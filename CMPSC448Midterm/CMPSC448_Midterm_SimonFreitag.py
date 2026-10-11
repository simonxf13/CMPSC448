"""
CMPSC 448 Midterm Project

Classify which LLM produced a response using a Convolutional Neural Network (CNN) and a Long Short-Term Memory network (LSTM).
For RQ2, test three kinds of text: input only, output only, and input + output.
"""

from collections import Counter

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Training settings
BATCH_SIZE = 64 # number of training examples processed before the model updates its weights
CNN_EPOCHS = 5 # number of training epochs for the CNN model
LSTM_EPOCHS = 15 # number of training epochs for the LSTM model
LEARNING_RATE = 0.01 # how large each weight update is during training
VOCAB_SIZE = 20000 # maximum number of words/tokens the model keeps in its vocab
EMBED_SIZE = 64 # number of values used to represent each word
HIDDEN_SIZE = 64 # size of the learned internal representation in CNN/LSTM

# Maximum number of words kept from each prompt/response
INPUT_MAX_LENGTH = 100
OUTPUT_MAX_LENGTH = 150

# Four LLM families used as the classification labels
LABELS = ["GPT", "Claude", "Llama", "Gemini"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

# Special vocabulary tokens
PAD = "<PAD>" # token used for padding sequences to the same length
UNK = "<UNK>" # token used for unknown words not in the vocabulary

# Local PersonalLLM files
TRAIN_FILE = "data/train.parquet"
TEST_FILE = "data/test.parquet"

# Make the run repeatable (controls the randomness)
SEED = 42
torch.manual_seed(SEED)

# Use a GPU if CUDA is available; otherwise use the CPU (GPU speeds up performance)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Map the full model names in PersonalLLM to the four labels used in the code
def get_label(model_name):
    name = str(model_name).lower()

    if name.startswith("openai/gpt-4o"):
        return "GPT"
    if name.startswith("anthropic/claude-3-sonnet"):
        return "Claude"
    if name.startswith("meta-llama/llama-3-70b-instruct"):
        return "Llama"
    if name.startswith("google/gemini-pro-1.5"):
        return "Gemini"

    return None


# Turn each original prompt into four examples: one for each selected LLM.
def convert_data(split):
    data = []

    for row in split:
        responses = {}

        # Each prompt can contain up to eight model responses.
        for i in range(1, 9):
            label = get_label(row.get(f"response_{i}_model"))
            response = row.get(f"response_{i}")

            # Keep only valid responses from the four selected model labels
            if label and isinstance(response, str) and response.strip():
                responses[label] = response

        # Only keep a prompt if it has responses from all four selected model labels
        if len(responses) == len(LABELS):
            for label in LABELS:
                data.append({"label": label, "input": str(row["prompt"]), "output": responses[label]})

    return data


# Load the provided train/test files and convert them to the project format.
def load_data():
    dataset = load_dataset("parquet", data_files={"train": TRAIN_FILE, "test": TEST_FILE})

    train_data = convert_data(dataset["train"])
    test_data = convert_data(dataset["test"])

    print(f"Training examples: {len(train_data)} | Testing examples: {len(test_data)}")
    return train_data, test_data


# Split text into words using whitespace tokenization
def tokenize(text):
    return str(text).split()


# Return the words to be used for the selected mode (input, output, or input_output)
def get_words(example, mode):
    input_words = tokenize(example["input"])[:INPUT_MAX_LENGTH]
    output_words = tokenize(example["output"])[:OUTPUT_MAX_LENGTH]

    if mode == "input":
        return input_words
    if mode == "output":
        return output_words
    if mode == "input_output":
        return input_words + output_words

    raise ValueError("Unknown mode")


# Return the padded length for the selected experiment mode
def get_max_length(mode):
    if mode == "input":
        return INPUT_MAX_LENGTH
    if mode == "output":
        return OUTPUT_MAX_LENGTH
    if mode == "input_output":
        return INPUT_MAX_LENGTH + OUTPUT_MAX_LENGTH

    raise ValueError("Unknown mode")


# Build a vocabulary from training data only
def build_vocab(train_data, mode):
    # Count the occurrences of each word in the training data
    counts = Counter()

    for example in train_data:
        counts.update(get_words(example, mode))

    # 0 is padding and 1 is used for words not seen during training.
    vocab = {PAD: 0, UNK: 1}

    # Add the most common words to the vocabulary (-2 to account for PAD and UNK) to ensure size of vocab remains at most VOCAB_SIZE
    for word, _ in counts.most_common(VOCAB_SIZE - 2):
        # generates vocabulary id for the current word, where each word gets assigned a unique ID sequentially as it is encountered, as it scales with length of the vocabulary
        vocab[word] = len(vocab)

    return vocab


# Convert text examples into fixed-length tensors that PyTorch can batch.
class TextDataset(Dataset):
    def __init__(self, data, vocab, mode):
        self.examples = []
        max_length = get_max_length(mode)

        # Each example in the dataset passed in as "label", "input", and "output" fields.
        for example in data:
            # Extract the words for the current example based on the selected mode
            words = get_words(example, mode)

            # Make sure every example contains at least one token (use UNK if empty).
            if not words:
                words = [UNK]

            # Replace each word with its vocabulary ID.
            ids = [vocab.get(word, vocab[UNK]) for word in words]
            
            # save the true number of words before passing, used by LSTM to find the final real word
            length = len(ids)

            # Pad every example to the same length.
            ids += [vocab[PAD]] * (max_length - length)

            # store the padded word-ID tensor, actual number of words, and the numerical label for the example LLM model
            self.examples.append((torch.tensor(ids, dtype=torch.long), length, LABEL_TO_ID[example["label"]]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


"""
Convolutional Neural Network:
Takes the word IDs, turns them into embeddings, search for useful word patterns with convolution, keeps the strongest detected patterns via max pooling,
and uses those patters to predict which LLM generated the text.

Follows the sequence given in the course: embedding -> convolution -> ReLU -> max pooling -> output
"""
class TextCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # convert each vocal ID into an EMBED_SIZE-dimensional vector, where <PAD> token uses 0 embedding
        self.embedding = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx=0)
        
        # apply HIDDEN_SIZE-dimensional filters across groups of 3 neighboring words, where each filter learns to recognize a local text pattern
        # so for "this is a model answer", will look at "this is a", "is a model", and "a model answer", etc.
        # Use Conv1d (1-dimension convolution layer)since we are looking at a 1-dimensional sequence of words.
        self.conv = nn.Conv1d(EMBED_SIZE, HIDDEN_SIZE, kernel_size=3)
        
        # Convert the HIDDEN_SIZE learned features into one score for each LLM class
        self.output = nn.Linear(HIDDEN_SIZE, len(LABELS))

    def forward(self, text, lengths):
        # Embedding output is [batch, words, embedding size].
        embeddings = self.embedding(text)

        # Pytorch Conv1d expects [batch, embedding size, words].
        embeddings = embeddings.transpose(1, 2)
        conv_features = torch.relu(self.conv(embeddings))

        # Keep the strongest value found by each convolution filter.
        pooled_features = torch.max(conv_features, dim=2).values
        return self.output(pooled_features)


"""
Long Short-Term Memory Network:
Takes the word IDs, turns them into embeddings, processes the text one word at a time with an LSTM,
and uses the final hidden state corresponding to the last non-padding word to predict which LLM generated the text.

Follows the sequence given in the course: embedding -> LSTM -> final real word -> output
"""
class TextLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # convert each vocal ID into an EMBED_SIZE-dimensional vector, where <PAD> token uses 0 embedding
        self.embedding = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx=0)
        # Process sequence one word at a time while keeping information from earlier words in the hidden state
        self.lstm = nn.LSTM(EMBED_SIZE, HIDDEN_SIZE, batch_first=True)
        # convert the final LSTM representation into one score for each LLM class
        self.output = nn.Linear(HIDDEN_SIZE, len(LABELS))

    def forward(self, text, lengths):
        # convert word IDs into embedding vectors
        embeddings = self.embedding(text)
        # process all embeddings through LSTM
        lstm_outputs, _ = self.lstm(embeddings)

        # Create one row for each example in the batch
        batch_indices = torch.arange(lstm_outputs.size(0), device=lstm_outputs.device)
        # get the LSTM output corresponding to the last non-padding word for each example
        final_outputs = lstm_outputs[batch_indices, lengths - 1]

        return self.output(final_outputs)


"""
Send training batches through CNN or LSTM, measure the loss, calculate how model's weights should change,
and update those weights to reduce the loss.

Uses cross-entropy loss and stochastic gradient descent for optimization.
"""
def train_model(model, train_loader, epochs):
    # Measure how wrong the model's predictions are
    loss_function = nn.CrossEntropyLoss()
    
    # Update model's weights using stochastic gradient descent (calculates error gradient for random small batches, updates weights immediately)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Go through entire training dataset once per epoch
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # train on one batch at a time
        for text, lengths, labels in train_loader:
            # Move this batch to the GPU if one is available.
            text = text.to(DEVICE)
            lengths = lengths.to(DEVICE)
            labels = labels.to(DEVICE)

            # Zero out the gradients from the previous step
            optimizer.zero_grad()
            
            # Send this batch through CNN or LSTM
            predictions = model(text, lengths)
            # Compare predictions with the correct labels
            loss = loss_function(predictions, labels)
            # Calculate how each model weight affected the loss
            loss.backward()
            # Update the weights in the direction that reduces the loss
            optimizer.step()
            # Accumulate the loss for this batch into the total loss for the epoch
            total_loss += loss.item()

        # Print average loss across all batches in the epoch
        print(f"  Epoch {epoch + 1}: loss = {total_loss / len(train_loader):.4f}")


# Use the trained model to make predictions on the test dataset, count how many predictions are correct, and return accuracy
def test_model(model, test_loader):
    # Tell Pytorch to switch model into evaluation mode
    model.eval()
    
    correct = 0
    total = 0
    
    # Disable gradients for testing to save memory and computations
    with torch.no_grad():
        # Iterate over the test dataset one batch at a time
        for text, lengths, labels in test_loader:
            # Move this batch to the GPU if one is available.
            text = text.to(DEVICE)
            lengths = lengths.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Send this batch through CNN or LSTM
            predictions = model(text, lengths)
            # Choose class with highest score
            predicted_labels = torch.argmax(predictions, dim=1)
            
            # Count how many predictions were correct and how many examples were tested
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

    # Accuracy is given by correct predictions / total predictions
    return correct / total

"""
RQ3: Generalization across subsets/domains
Convert the data again while also keeping the PersonalLLM subset.
The subset is used as a task/source group for the RQ3 generalization test.
"""
def convert_domain_data(split):
    data = []

    for row in split:
        responses = {}

        # Each prompt can contain up to eight model responses.
        for i in range(1, 9):
            label = get_label(row.get(f"response_{i}_model"))
            response = row.get(f"response_{i}")

            # Keep only valid responses from the four selected model labels.
            if label and isinstance(response, str) and response.strip():
                responses[label] = response

        # Only keep a prompt if it has responses from all four selected model labels.
        if len(responses) == len(LABELS):
            for label in LABELS:
                data.append({"label": label, "input": str(row["prompt"]), "output": responses[label], "subset": str(row["subset"])})

    return data

# Train and test one model for the RQ3 experiment.
def test_unseen_subset(Model, name, train_data, test_data):
    mode = "output"

    # Build the vocabulary using only the RQ3 training data.
    vocab = build_vocab(train_data, mode)

    train_dataset = TextDataset(train_data, vocab, mode)
    test_dataset = TextDataset(test_data, vocab, mode)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = Model(len(vocab)).to(DEVICE)

    print(f"\n  {name}")

    if name == "CNN":
        train_model(model, train_loader, CNN_EPOCHS)
    else:
        train_model(model, train_loader, LSTM_EPOCHS)

    return test_model(model, test_loader)


# Test whether CNN and LSTM can identify LLMs on a subset that was not used for training.
def run_rq3(results):
    print("\nRQ3 - Generalization Across Subsets")

    # Reload the original rows because the normal project data does not keep subset.
    dataset = load_dataset("parquet", data_files={"train": TRAIN_FILE, "test": TEST_FILE})

    # RQ3 requires the PersonalLLM subset field.
    if "subset" not in dataset["train"].column_names or "subset" not in dataset["test"].column_names:
        print("Dataset does not contain a subset column.")
        return

    train_data = convert_domain_data(dataset["train"])
    test_data = convert_domain_data(dataset["test"])

    # Count the training examples in each subset.
    train_subset_counts = Counter()

    for example in train_data:
        subset = example["subset"]
        train_subset_counts[subset] += 1

    # Count the testing examples in each subset.
    test_subset_counts = Counter()

    for example in test_data:
        subset = example["subset"]
        test_subset_counts[subset] += 1

    # Choose the subset with the most test examples that also appears in training.
    kept_subset = None
    largest_test_count = 0

    for subset in test_subset_counts:
        if subset in train_subset_counts:
            if test_subset_counts[subset] > largest_test_count:
                kept_subset = subset
                largest_test_count = test_subset_counts[subset]

    if kept_subset is None:
        print("No subset appears in both train and test data, RQ3 skipped.")
        return

    # Remove the kept subset completely from the RQ3 training data.
    rq3_train_data = []

    for example in train_data:
        if example["subset"] != kept_subset:
            rq3_train_data.append(example)

    # Test only on examples from the kept subset.
    rq3_test_data = []

    for example in test_data:
        if example["subset"] == kept_subset:
            rq3_test_data.append(example)

    print(f"Kept subset: {kept_subset}")
    print(f"Training examples without kept subset: {len(rq3_train_data)}")
    print(f"Testing examples from kept subset: {len(rq3_test_data)}")

    # Train the CNN without the kept subset.
    cnn_accuracy = test_unseen_subset(TextCNN, "CNN", rq3_train_data, rq3_test_data)

    # Train the LSTM without the kept subset.
    lstm_accuracy = test_unseen_subset(TextLSTM, "LSTM", rq3_train_data, rq3_test_data)

    # Find the original output-only accuracies from RQ1.
    original_cnn_accuracy = 0
    original_lstm_accuracy = 0

    for name, mode, accuracy in results:
        if name == "CNN" and mode == "output":
            original_cnn_accuracy = accuracy

        if name == "LSTM" and mode == "output":
            original_lstm_accuracy = accuracy

    print("\nRQ3 Comparison")
    print(f"CNN normal output accuracy: {original_cnn_accuracy:.4f}")
    print(f"CNN unseen-subset accuracy: {cnn_accuracy:.4f}")
    print(f"LSTM normal output accuracy: {original_lstm_accuracy:.4f}")
    print(f"LSTM unseen-subset accuracy: {lstm_accuracy:.4f}")
        

"""
RQ4: Characteristics that distinguish LLMs

Tests classification accuracy of LLM responses based on linguistic and structural characteristics.
Characteristics include average words, vocabulary size, lexical diversity, and average formatting tokens.
Characteristics chosen based on recommendations from assignment instructions.
"""
# Print simple linguistic and structural measurements for each LLM.
def analyze_response_characteristics(data):
    print("\nRQ4 - Response Characteristics")

    for label in LABELS:
        response_count = 0
        total_words = 0
        diversity_total = 0
        formatting_total = 0
        unique_words = set()

        for example in data:
            if example["label"] == label:
                words = tokenize(example["output"])[:OUTPUT_MAX_LENGTH]

                response_count += 1
                total_words += len(words)

                # Add each word to the set of unique words used by this LLM.
                for word in words:
                    unique_words.add(word)

                    # Count simple Markdown/formatting tokens.
                    if word.startswith("*"):
                        formatting_total += 1
                    elif word.startswith("#"):
                        formatting_total += 1
                    elif word.startswith("`"):
                        formatting_total += 1
                    elif word.startswith(">"):
                        formatting_total += 1
                    elif word.startswith("-"):
                        formatting_total += 1

                # Lexical diversity is unique words divided by total words within one response.
                if len(words) > 0:
                    response_unique_words = set(words)
                    diversity_total += len(response_unique_words) / len(words)

        if response_count > 0:
            average_length = total_words / response_count
            average_diversity = diversity_total / response_count
            average_formatting = formatting_total / response_count
        else:
            average_length = 0
            average_diversity = 0
            average_formatting = 0

        print(f"\n{label}")
        print(f"Average words: {average_length:.2f}")
        print(f"Vocabulary size: {len(unique_words)}")
        print(f"Lexical diversity: {average_diversity:.4f}")
        print(f"Average formatting tokens: {average_formatting:.2f}")


# Remove simple formatting symbols while leaving the words themselves for testing
def remove_formatting(text):
    cleaned_text = str(text)

    symbols = ["*", "#", "`", ">", "-"]

    for symbol in symbols:
        cleaned_text = cleaned_text.replace(symbol, " ")

    return cleaned_text

# Make a copy of the dataset with formatting removed from each LLM output.
def remove_formatting_from_data(data):
    cleaned_data = []

    for example in data:
        cleaned_example = example.copy()
        cleaned_example["output"] = remove_formatting(example["output"])
        cleaned_data.append(cleaned_example)

    return cleaned_data

# Train and test one model without formatting.
def test_without_formatting(Model, name, train_data, test_data):
    mode = "output"

    vocab = build_vocab(train_data, mode)

    train_dataset = TextDataset(train_data, vocab, mode)
    test_dataset = TextDataset(test_data, vocab, mode)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = Model(len(vocab)).to(DEVICE)

    print(f"\n  {name}")

    if name == "CNN":
        train_model(model, train_loader, CNN_EPOCHS)
    else:
        train_model(model, train_loader, LSTM_EPOCHS)

    return test_model(model, test_loader)


# Compare output-only classification before and after removing formatting signals.
def run_rq4(train_data, test_data, results):
    # First measure response length, vocabulary, lexical diversity, and formatting.
    analyze_response_characteristics(test_data)

    # Remove formatting from copies of both the training and testing data.
    cleaned_train_data = remove_formatting_from_data(train_data)
    cleaned_test_data = remove_formatting_from_data(test_data)

    print("\nRQ4 - Formatting Removal")

    # Train the CNN after removing formatting.
    cnn_accuracy = test_without_formatting(TextCNN,"CNN",cleaned_train_data,cleaned_test_data)

    # Train the LSTM after removing formatting.
    lstm_accuracy = test_without_formatting(TextLSTM,"LSTM",cleaned_train_data,cleaned_test_data)

    # Find the original output-only accuracies from RQ1.
    original_cnn_accuracy = 0
    original_lstm_accuracy = 0

    for name, mode, accuracy in results:
        if name == "CNN" and mode == "output":
            original_cnn_accuracy = accuracy

        if name == "LSTM" and mode == "output":
            original_lstm_accuracy = accuracy

    print("\nRQ4 Comparison")
    print(f"CNN original output accuracy: {original_cnn_accuracy:.4f}")
    print(f"CNN without formatting: {cnn_accuracy:.4f}")
    print(f"LSTM original output accuracy: {original_lstm_accuracy:.4f}")
    print(f"LSTM without formatting: {lstm_accuracy:.4f}")



# Run CNN and LSTM on input only, output only, and input + output.
def main():
    train_data, test_data = load_data()
    print(f"Using device: {DEVICE}")
    results = []

    for mode in ["input", "output", "input_output"]:
        vocab = build_vocab(train_data, mode)
        train_dataset = TextDataset(train_data, vocab, mode)
        test_dataset = TextDataset(test_data, vocab, mode)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        print(f"\n{mode} | vocabulary size = {len(vocab)}")

        # Train and test both model types.
        for name, Model in [("CNN", TextCNN), ("LSTM", TextLSTM)]:
            model = Model(len(vocab)).to(DEVICE)
            print(name)
            # Use more epochs for LSTM as it takes more training to yield better results
            if name == "CNN":
                train_model(model, train_loader, CNN_EPOCHS)
            else:
                train_model(model, train_loader, LSTM_EPOCHS)
            accuracy = test_model(model, test_loader)
            print(f"Test accuracy = {accuracy:.4f}")
            results.append((name, mode, accuracy))

    print("\nFinal Results")
    for name, mode, accuracy in results:
        print(f"{name:4} | {mode:12} | {accuracy:.4f}")
        
    # Run the extra-credit RQ3
    run_rq3(results)

    # Run the extra-credit RQ4
    run_rq4(train_data, test_data, results)


if __name__ == "__main__":
    main()
