#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


os.chdir("D:\Courses\CSC 5991 Large Language Model (LLM)\Attention_CNN")


# In[6]:


# Download Flickr8k Dataset (Images and Captions)
Flickr8k_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
Flickr8k_Captions_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

if not os.path.exists("Flickr8k_Dataset.zip"):
    print("Downloading Flickr8k Images...")
    r = requests.get(Flickr8k_URL, stream=True)
    with open("Flickr8k_Dataset.zip", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)

if not os.path.exists("Flickr8k_text.zip"):
    print("Downloading Flickr8k Captions...")
    r = requests.get(Flickr8k_Captions_URL, stream=True)
    with open("Flickr8k_text.zip", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)


# In[7]:


# Extract Dataset
with zipfile.ZipFile("Flickr8k_Dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("./")
with zipfile.ZipFile("Flickr8k_text.zip", 'r') as zip_ref:
    zip_ref.extractall("./")


# In[8]:


# Load Captions
caption_file = "D:/Courses/CSC 5991 Large Language Model (LLM)/Attention_CNN/Flickr8k.token.txt"
image_folder = "D:/Courses/CSC 5991 Large Language Model (LLM)\Attention_CNN/Flicker8k_Dataset"

captions = {}
with open(caption_file, 'r') as f:
    for line in f:
        parts = line.strip().split("#")
        img_id = parts[0]
        caption = parts[1].split("\t")[1].lower()
        if img_id not in captions:
            captions[img_id] = []
        captions[img_id].append(caption)


# In[9]:


# Preprocessing Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# In[10]:


# Load MobileNet for Feature Extraction
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet = mobilenet.features.to(device)
mobilenet.eval()


# In[11]:


def extract_features(image_path):
    """Extract features from an image using MobileNet."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = mobilenet(image)  # Expected Shape: [1, 1280, 7, 7]

    # Ensure tensor is contiguous and correctly reshaped
    batch_size, feature_dim, height, width = features.shape  # [1, 1280, 7, 7]
    features = features.contiguous().reshape(batch_size, feature_dim, height * width).permute(0, 2, 1)  # [1, 49, 1280]

    print("Feature Shape After Reshape:", features.shape)  # Debugging Print
    return features


# In[12]:


# Sample extraction
sample_img = list(captions.keys())[0]
sample_features = extract_features(os.path.join(image_folder, sample_img))
print("Feature shape:", sample_features.shape)


# In[13]:


# Caption Preprocessing
import nltk
nltk.download('punkt')
def tokenize_caption(caption):
    return nltk.tokenize.word_tokenize(caption.lower())

word_freq = Counter()
for img_id, caps in captions.items():
    for cap in caps:
        word_freq.update(tokenize_caption(cap))


# In[14]:


# Create Vocabulary
vocab = {word: idx+1 for idx, (word, _) in enumerate(word_freq.most_common())}
vocab["<PAD>"] = 0
vocab["<SOS>"] = len(vocab) + 1
vocab["<EOS>"] = len(vocab) + 2

# Create Index-to-Word Mapping
idx_to_word = {idx: word for word, idx in vocab.items()}


# In[15]:


# Convert Captions to Indices
def caption_to_indices(caption, vocab, max_length=20):
    tokens = tokenize_caption(caption)
    indices = [vocab.get(word, vocab["<PAD>"]) for word in tokens]  # Use PAD token if word not found
    indices = [vocab["<SOS>"]] + indices[:max_length-2] + [vocab["<EOS>"]]
    indices += [vocab["<PAD>"]] * (max_length - len(indices))

    # Ensure all indices are within vocab range
    indices = [min(idx, len(vocab) - 1) for idx in indices]
    return indices


# In[16]:


# Define Dataset Class
class Flickr8kDataset(Dataset):
    def __init__(self, image_folder, captions, vocab):
        self.image_folder = image_folder
        self.captions = captions
        self.vocab = vocab
        self.image_ids = list(self.captions.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_folder, img_id)

        # Check if file exists, else return None
        if not os.path.exists(img_path):
            return None  # Instead of crashing, return None

        image_features = extract_features(img_path)
        caption = caption_to_indices(self.captions[img_id][0], self.vocab)

        return image_features.clone().detach(), torch.tensor(caption).clone().detach()


# In[17]:


print(type(captions))  # Should print <class 'dict'>


# In[18]:


# Create DataLoader
dataset = Flickr8kDataset(image_folder, captions, vocab)
def collate_fn(batch):
    """Remove None values from the batch."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # If the whole batch is None, return None
    return torch.utils.data.dataloader.default_collate(batch)

# Use the new collate function in DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


# In[19]:


# Get a batch of data from DataLoader
for img_features, captions in dataloader:
    print("Batch Image Feature Shape:", img_features.shape)  # Expected: [batch_size, 1280, 7, 7]
    print("Batch Caption Shape:", captions.shape)  # Expected: [batch_size, max_caption_length]

    # Print a sample image feature vector and its corresponding caption indices
    print("\nSample Image Feature Vector:", img_features[0, :, :, :].shape)
    print("Sample Caption Indices:", captions[0].tolist())

    # Decode back to words (just for checking)
    decoded_caption = [idx_to_word[idx] for idx in captions[0].tolist() if idx in idx_to_word and idx not in {vocab["<PAD>"]}]
    print("Decoded Caption:", " ".join(decoded_caption))  # Should print a readable caption

    break  # Only print first batch


# In[20]:


class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, features, hidden):
        """
        features: Image feature map from MobileNet (Expected: [batch_size, 49, 1280])
        hidden: Decoder hidden state [1, batch_size, hidden_dim]
        """

        # Debugging print
        print("Actual Feature Shape Before Fix:", features.shape)  # Should be [batch_size, 49, 1280]

        batch_size, seq_length, feature_dim = features.shape  # [batch_size, 49, 1280]

        # ✅ Fix: Ensure `hidden` has the correct shape
        if hidden.dim() == 3 and hidden.shape[0] == 1:
            hidden = hidden.squeeze(0)  # Shape: [batch_size, hidden_dim]

        hidden = hidden.unsqueeze(1).expand(-1, seq_length, -1)  # Shape: [batch_size, 49, 512]

        print("Hidden Shape After Fix:", hidden.shape)  # Should be [batch_size, 49, 512]

        # Concatenate features and hidden state
        concat_features = torch.cat((features, hidden), dim=2)  # [batch_size, 49, 1280+512]

        # Compute attention scores
        attention_scores = self.v(torch.tanh(self.attn(concat_features)))  # Shape: [batch_size, 49, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # Normalize over spatial locations

        # Compute context vector
        context_vector = (attention_weights * features).sum(dim=1)  # Shape: [batch_size, 1280]

        return context_vector, attention_weights


# In[21]:


class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, feature_dim):
        super(DecoderGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(feature_dim, hidden_size)
        self.gru = nn.GRU(embed_size + feature_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, hidden):
        """
        features: Image features [batch_size, 1280, 7, 7]
        captions: Tokenized caption indices [batch_size, seq_length]
        hidden: Decoder hidden state [1, batch_size, hidden_size]
        """
        batch_size, seq_length = captions.shape

        # Embed captions
        embeddings = self.embedding(captions)  # Shape: [batch_size, seq_length, embed_size]

        # Generate context vector from attention (per timestep)
        context_vector, _ = self.attention(features, hidden.squeeze(0))  # Shape: [batch_size, 1280]

        # Expand context vector to match the sequence length
        context_vector = context_vector.unsqueeze(1).repeat(1, seq_length, 1)  # Shape: [batch_size, seq_length, 1280]

        # Concatenate embeddings and context vector correctly
        gru_input = torch.cat((embeddings, context_vector), dim=2)  # Shape: [batch_size, seq_length, embed_size + 1280]

        # Pass through GRU
        outputs, hidden = self.gru(gru_input, hidden)  # GRU output: [batch_size, seq_length, hidden_size]

        # Fully connected layer to predict words
        outputs = self.fc(outputs)  # Shape: [batch_size, seq_length, vocab_size]

        return outputs, hidden


# In[22]:


# Define model hyperparameters
vocab_size = len(vocab)
embed_size = 256  # Embedding dimension
hidden_size = 512  # GRU hidden size
feature_dim = 1280  # MobileNet feature dimension

# Initialize Decoder
decoder = DecoderGRU(vocab_size, embed_size, hidden_size, feature_dim).to(device)


# In[45]:


# Define Test Function for Decoder
def generate_caption(encoder, decoder, image_path, vocab, max_length=20):
    encoder.eval()
    decoder.eval()

    # Extract image features
    image_features = extract_features(image_path).unsqueeze(0).to(device)
    hidden = torch.zeros(1, 1, decoder.gru.hidden_size).to(device) # Remove one dimension from the hidden state

    caption = [vocab["<SOS>"]]
    for _ in range(max_length):
        caption_tensor = torch.tensor([caption[-1]]).unsqueeze(0).to(device)
        output, hidden = decoder(image_features, caption_tensor, hidden)
        predicted_idx = output.argmax(dim=-1).item()
        caption.append(predicted_idx)
        if predicted_idx == vocab["<EOS>"]:
            break

    # Convert indices to words
    decoded_caption = [idx_to_word[idx] for idx in caption if idx in idx_to_word]
    return " ".join(decoded_caption)

print("Decoder test function added successfully!")


# In[ ]:


test_image_path = "/content/Flicker8k_Dataset/1001773457_577c3a7d70.jpg"  # Replace with an actual image path
generated_caption = generate_caption(mobilenet, decoder, test_image_path, vocab)
print("Generated Caption:", generated_caption)


# In[25]:


# Define Training Function
def train_model(encoder, decoder, dataloader, vocab_size, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD tokens
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    encoder.to(device)
    decoder.to(device)

    decoder.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for img_features, captions in tqdm(dataloader):
            img_features, captions = img_features.to(device), captions.to(device)  # Move to device

            optimizer.zero_grad()
            hidden = torch.zeros(1, img_features.size(0), decoder.gru.hidden_size).to(device)  # Move hidden to device

            outputs, _ = decoder(img_features, captions[:, :-1], hidden)  # Predict all words except last
            loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))  # Compare to all words except first

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print("Training complete!")


# In[26]:


image_folder = "D:/Courses/CSC 5991 Large Language Model (LLM)\Attention_CNN/Flicker8k_Dataset"  # Ensure correct path
image_files = os.listdir(image_folder)

print(f"Total images: {len(image_files)}")
print(f"Example images: {image_files[:10]}")


# In[ ]:


num_epochs = 10  # You can increase this if needed
learning_rate = 0.001
vocab_size = len(vocab)

train_model(mobilenet, decoder, dataloader, vocab_size, num_epochs, learning_rate)


# In[ ]:


for img_features, captions in dataloader:
    captions = captions.to(device)
    print("Max index in captions:", captions.max().item())
    print("Vocab size:", vocab_size)
    break  # Check first batch only


# In[ ]:


def generate_caption(encoder, decoder, image_path, vocab, max_length=20):
    encoder.eval()
    decoder.eval()

    # Extract image features
    image_features = extract_features(image_path).unsqueeze(0).to(device)  # Ensure batch dimension

    # ✅ Fix: Ensure correct feature shape
    print("Feature Shape Before Fix in generate_caption:", image_features.shape)
    if image_features.dim() == 4:  # If it has an extra dimension, remove it
        image_features = image_features.squeeze(1)
    print("Feature Shape After Fix in generate_caption:", image_features.shape)

    # Initialize hidden state
    hidden = torch.zeros(1, 1, decoder.gru.hidden_size).to(device)

    caption = [vocab["<SOS>"]]
    for _ in range(max_length):
        caption_tensor = torch.tensor([caption[-1]]).unsqueeze(0).to(device)

        # Predict next word
        output, hidden = decoder(image_features, caption_tensor, hidden)
        predicted_idx = output.argmax(dim=-1).item()

        caption.append(predicted_idx)
        if predicted_idx == vocab["<EOS>"]:
            break

    # Convert indices to words
    idx_to_word = {idx: word for word, idx in vocab.items()}
    decoded_caption = [idx_to_word[idx] for idx in caption if idx in idx_to_word]

    return " ".join(decoded_caption)


# In[ ]:


test_image_path = "D:/Courses/CSC 5991 Large Language Model (LLM)\Attention_CNN/Flicker8k_Dataset/1032460886_4a598ed535.jpg"  # Replace with actual path
generated_caption = generate_caption(mobilenet, decoder, test_image_path, vocab)
print("Generated Caption:", generated_caption)


# In[ ]:


with torch.no_grad():
    model.eval()
    train_bleu = evaluate_model(desc=f'Train: ', model=final_model,
                                loss_fn=loss_fn, bleu_score_fn=corpus_bleu_score_fn,
                                tensor_to_word_fn=tensor_to_word_fn,
                                data_loader=train_eval_loader, vocab_size=vocab_size)
    val_bleu = evaluate_model(desc=f'Val: ', model=final_model,
                              loss_fn=loss_fn, bleu_score_fn=corpus_bleu_score_fn,
                              tensor_to_word_fn=tensor_to_word_fn,
                              data_loader=val_loader, vocab_size=vocab_size)
    test_bleu = evaluate_model(desc=f'Test: ', model=final_model,
                               loss_fn=loss_fn, bleu_score_fn=corpus_bleu_score_fn,
                               tensor_to_word_fn=tensor_to_word_fn,
                               data_loader=test_loader, vocab_size=vocab_size)
    for setname, result in zip(('train', 'val', 'test'), (train_bleu, val_bleu, test_bleu)):
        print(setname, end=' ')
        for ngram in (1, 2, 3, 4):
            print(f'Bleu-{ngram}: {result[ngram]}', end=' ')
        print()

