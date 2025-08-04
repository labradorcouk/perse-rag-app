import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from dotenv import load_dotenv


def fine_tune_embedding_model(df, base_model, output_dir, batch_size, epochs):
    print("Preparing training examples ...")
    if not {'question', 'answer'}.issubset(df.columns):
        raise ValueError("Data must contain 'question' and 'answer' columns.")
    train_examples = [
        InputExample(texts=[str(row['question']), str(row['answer'])], label=1.0)
        for _, row in df.iterrows()
    ]
    print(f"Loading base model: {base_model} ...")
    model = SentenceTransformer(base_model)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    print(f"Fine-tuning for {epochs} epochs ...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        show_progress_bar=True
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving fine-tuned model to {output_dir} ...")
    model.save(output_dir)
    print("Done!")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fine-tune embedding model using local Q&A CSV file.")
    parser.add_argument('--csv_path', type=str, default='../epcNonDomesticScotlandQA.csv', help='Path to local Q&A CSV file (default: ../epcNonDomesticScotlandQA.csv)')
    parser.add_argument('--output_dir', type=str, default='models/epc-ndscotland-finetuned', help='Directory to save fine-tuned model')
    parser.add_argument('--base_model', type=str, default='all-MiniLM-L6-v2', help='Base sentence-transformers model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    args = parser.parse_args()

    # Step 1: Load Q&A data from local CSV
    print(f"Loading Q&A data from {args.csv_path} ...")
    df = pd.read_csv(args.csv_path)

    # Step 2: Fine-tune embedding model
    fine_tune_embedding_model(df, args.base_model, args.output_dir, args.batch_size, args.epochs)

if __name__ == "__main__":
    main() 