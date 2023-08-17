import numpy as np
import pandas as pd
import torch

import torch.optim as optim
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim.lr_scheduler import StepLR


class CustomDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_len):
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = str(self.text[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


# 加载数据
df = pd.read_csv('output_utf8.csv')

# 对于Category列，我们需要将其转换为列表格式，因为每个样本可能有多个类别
df['Category'] = df['Category'].apply(lambda x: [x])

# 初始化MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# 对Category列进行二值化处理
df['Category'] = list(mlb.fit_transform(df['Category']))

# 定义文本和标签
texts = df['Description'].tolist()
labels = df['Category'].tolist()

# 设定预训练模型名称
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

# 初始化tokenizer
tokenizer = BertTokenizerFast.from_pretrained(PRE_TRAINED_MODEL_NAME)

# 设定最大序列长度
MAX_LEN = 256

# 创建数据集
dataset = CustomDataset(texts, labels, tokenizer, MAX_LEN)

# 按照一定比例拆分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
BATCH_SIZE = 64  # 增加batch size
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型
model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=len(mlb.classes_))

# 使用GPU进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 设置优化器和损失函数
optimizer = optim.RMSprop(model.parameters(), lr=1e-5)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 添加学习率调度
loss_fn = CrossEntropyLoss()

# 设置训练的轮数
EPOCHS = 100

# 设置梯度累积的步数
gradient_accumulation_steps = 2


# 定义评估模型的函数
def evaluate_model(model, dataloader):
    model.eval()

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].detach().cpu().numpy()

            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)

            true_labels.append(labels)
            pred_labels.append(preds)

    true_labels_flat = np.vstack(true_labels)
    pred_labels_flat = np.vstack(pred_labels)

    # Print classification report
    print(classification_report(true_labels_flat, pred_labels_flat, target_names=list(map(str, mlb.classes_))))

    # Compute micro/macro-average precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_flat, pred_labels_flat, average='micro')
    print(f"Micro-average Precision: {precision:.4f}")
    print(f"Micro-average Recall: {recall:.4f}")
    print(f"Micro-average F1 Score: {f1:.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_flat, pred_labels_flat, average='macro')
    print(f"Macro-average Precision: {precision:.4f}")
    print(f"Macro-average Recall: {recall:.4f}")
    print(f"Macro-average F1 Score: {f1:.4f}")


# 开始训练

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    model.train()

    for i, data in enumerate(tqdm(train_loader, total=len(train_loader))):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    evaluate_model(model, val_loader)

# 保存模型
model.save_pretrained("my_model3000")
