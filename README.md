# Lipophilicity Prediction using MoLFormer-XL

This project fine-tunes the `ibm/MoLFormer-XL-both-10pct` model to predict **lipophilicity** (logD) from molecular SMILES strings. We apply various techniques, including regression adaptation, unsupervised pretraining, influence-based data selection, and parameter-efficient fine-tuning.

---

## Dataset

- **Original Dataset**: [MoleculeNet Lipophilicity](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_Lipophilicity) (4,200 compounds)
- **External Dataset**: 12,000 compounds (used for influence-based augmentation)
- **Split**: 80% training, 20% testing

---

## Tasks Overview
![image](https://github.com/user-attachments/assets/54961ab9-905f-43c9-8b67-6e14b6edba88)

### Task 1: Baseline & Supervised Fine-Tuning

#### Objective:
Adapt MoLFormer-XL for regression tasks to predict lipophilicity.

#### Steps:
- Load and tokenize the Lipophilicity dataset using a custom `SMILESDataset` class.
- Attach a regression head (linear layer) to the pretrained model.
- Fine-tune using supervised learning.

#### Training Details:
- Batch Size: 16
- Learning Rate: 1e-5 (AdamW)
- Epochs: 15 with early stopping
  
![image](https://github.com/user-attachments/assets/8864bf0b-9ed4-4518-a86e-4534da7864c9)

#### Evaluation Metrics:
- MSE: 0.4507
- R²: 0.7019
- RMSE: 0.6637

---

### Task 1.2: Unsupervised Fine-Tuning (MLM)

#### Objective:
Enhance chemical understanding via unsupervised **Masked Language Modeling (MLM)** on SMILES strings.

- MLM fine-tuning applied using the original training set (no labels).
- Improved downstream regression performance after retraining the regression head.
![image](https://github.com/user-attachments/assets/7190a8ad-3e2b-42a6-adbe-e35bd693e708)
![image](https://github.com/user-attachments/assets/816798a1-85c7-49b6-ab00-ed82b3da5941)

| Metric          | Original Model | Fine-Tuned Model |
| --------------- | -------------- | ---------------- |
| Test Loss (MSE) | 0.4652         | 0.4292           |
| R² Score        | 0.6927         | 0.7134           |
| RMSE            | 0.6738         | 0.6507           |


---

### Task 2: Influence-Based Data Augmentation

#### Objective:
Improve model performance by adding selected high-impact samples from an external dataset.

#### Key Steps:
1. **Compute Influence Scores** using LiSSA (approximated inverse Hessian-vector product).
2. **Top 100 External Samples** identified as most impactful.
3. **Augment Training Data** with these samples.
4. **Retrain the Model** and compare results to baseline.

#### Results:
![image](https://github.com/user-attachments/assets/68475c7b-b4bf-4985-92fe-dbca3d1a938e)

- Significant improvements in MSE and RMSE.
- Slight risk of overfitting detected — further validation recommended.

| Metric          | Before Augmentation | After Augmentation |
| --------------- | -------------- | ---------------- |
| Test Loss (MSE) |0.4244         | 0.0775          |
| R² Score        |0.7189       | 0.9477        |
| RMSE            | 0.6445        | 0.2781           |

---

### Task 3: Efficient Fine-Tuning with Data Selection

#### Objective:
Improve generalization using smart data selection and parameter-efficient tuning.

#### 3.1 Data Selection Strategies:
- **Random Sampling**: Unbiased subset selection.
- **Diversity Sampling**: K-means clustering over embeddings to select diverse representatives.
- **Uncertainty Sampling**: Select high-entropy samples where the model is least confident.
![image](https://github.com/user-attachments/assets/786ca7fa-ad6e-4773-bd87-ed8c9ce9c6dd)

|        | Loss (MSE)| R-square | RMSE |
| --------------- | -------------- | ---------------- | ------|
| Random | 0.4128     | 0.7250  | 0.6375  |
| Uncertainity  |0.4281   | 0.7145     |  0.6495  |
| Diverse    | 0.4220  | 0.7194     |  0.6439  |

#### 3.2 Fine-Tuning Techniques:
- **BitFit**: Fine-tune only bias terms (lightweight).
- **LoRA (Low-Rank Adaptation)**: Add low-rank matrices for efficient adaptation.
- **iA3 (Implicit Adapters)**: Plug adapters into select layers.



#### 3.3 Results:
- **LoRA** showed the best performance among all strategies, achieving:
  - Low MSE and RMSE
  - High R²
  - Strong generalization without overfitting
    
  ![image](https://github.com/user-attachments/assets/df7ef501-f70d-472e-8a25-6c5e6b63d6da)
 

|        | Loss (MSE)| R-square | RMSE |
| --------------- | -------------- | ---------------- | ------|
| LoRA | 0.3520    | 0.9512  | 0.2658  |
| IA3  |0.3673  | 0.9491  |  0.2716  |
| BitFit   | 0.3663  | 0.9492     |  0.2713  |

---

## Final Observations

- **Task 1** provides a solid supervised baseline.
- **Task 2** boosts performance through carefully curated external data.
- **Task 3** achieves efficient performance gains via smarter fine-tuning and sample selection.
- **LoRA** emerges as the most effective fine-tuning strategy overall.

|        | Task 1 (Fine-Tuning)| Task 2 (Influence) | Task 3 (LoRA) |
| --------------- | -------------- | ---------------- | ------|
| MSE | 0.4252    | 0.0771  | 0.3520  |
| R-Squared  |0.7165  | 0.9476  |  0.9512  |
| RMSE | 0.6471  | 0.2782    |  0.2658  |

![image](https://github.com/user-attachments/assets/fcea4c1d-7d68-4cf9-b887-9a58e3455661)

---

