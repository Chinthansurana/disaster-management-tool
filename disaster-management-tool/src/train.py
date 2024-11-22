import torch
from torch.utils.data import DataLoader, Dataset
from model import SiameseNetwork, EncoderDecoderDamageDetection
from data_processing import load_images_from_directory, preprocess_data, create_train_test_split

# Prepare dataset
images = load_images_from_directory('data/raw')
images = preprocess_data(images)
labels = np.random.randint(0, 2, size=(images.shape[0],))  # Placeholder labels
X_train, X_test, y_train, y_test = create_train_test_split(images, labels)

# DataLoader
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

train_loader = DataLoader(ImageDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(ImageDataset(X_test, y_test), batch_size=32)

# Instantiate models
siamese_model = SiameseNetwork().cuda()
encoder_decoder_model = EncoderDecoderDamageDetection().cuda()

# Optimizer and loss
optimizer = torch.optim.Adam(list(siamese_model.parameters()) + list(encoder_decoder_model.parameters()), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(10):
    siamese_model.train()
    encoder_decoder_model.train()
    for imgs, labels in train_loader:
        imgs = imgs.cuda()
        labels = labels.cuda()
        
        optimizer.zero_grad()
        
        # Forward pass (Siamese + Encoder-Decoder)
        output = siamese_model(imgs, imgs)  # Dummy pass for now
        loss = criterion(output, labels.float())
        
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
