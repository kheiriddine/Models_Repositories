!pip install -r requirements.txt
import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import json
import tempfile
import os
import torch
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
import plotly.express as px
import re
import time
from together import Together

# GitHub Configuration
GITHUB_USER = "kheiriddine"
GITHUB_REPO = "Models_Repositories"
GITHUB_BRANCH = "main"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents"

# File Paths
DATA_PATH = "gradiant_poc-code_base-02111a06fe82/gradiant_poc-code_base-02111a06fe82/0. Donnees Mock-up/Pharma_Procurement_Transactions.csv"
XGBOOST_MODEL_PATH = "model_fin_xgboost_classifier/xgboost_model.json"
LABEL_ENCODER_PATH = "model_fin_xgboost_classifier/label_encoder.json"
BERT_MODEL_DIR = "bert_model"
DISTILBERT_MODEL_DIR = "DistilBert_model"

# Initialize session states
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {
        'model': None,
        'embedder': None,
        'label_encoder': None
    }

if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None

# Configure page
st.set_page_config(page_title="Procurement Dashboard", layout="wide")
st.title("Procurement Transaction Analysis")

st.markdown("""
<style>
    /* Base styles */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6;
        color: #2c3e50;
    }
    #.stApp {
        #background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    #}
    unsafe_allow_html=True
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f3f4f9 0%, #e6e9f0 100%);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(5px);
    }
    
    /* Buttons - with hover animation */
    .stButton>button {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(46, 125, 50, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.4);
    }
    
    /* Metrics cards - with subtle animation */
    .stMetric {
        background: rgba(224, 247, 250, 0.8);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #4CAF50;
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: scale(1.02);
        background: rgba(224, 247, 250, 1);
    }
    
    /* Sidebar - glassmorphism effect */
    .sidebar .block-container {
        padding-top: 2rem;
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-radius: 0 12px 12px 0;
        box-shadow: 4px 0 15px rgba(0,0,0,0.05);
    }
    
    /* Upload box - with pulse animation */
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(248, 249, 250, 0.6);
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { border-color: #4CAF50; }
        50% { border-color: #81C784; }
        100% { border-color: #4CAF50; }
    }
    
    .upload-box:hover {
        background: rgba(248, 249, 250, 0.9);
        transform: translateY(-3px);
    }
    
    /* Headers with gradient text */
    h1, h2, h3 {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0f7fa;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(#4CAF50, #2E7D32);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_procurement_data():
    try:
        # Method 1: API approach (recommended)
        api_url = "https://api.github.com/repos/kheiriddine/Models_Repositories/contents/gradiant_poc-code_base-02111a06fe82/gradiant_poc-code_base-02111a06fe82/0.%20Donnees%20Mock-up/Pharma_Procurement_Transactions.csv"

        response = requests.get(api_url, headers={
            "Accept": "application/vnd.github.v3.raw",
            "Authorization": "token YOUR_GH_TOKEN"  # Only for private repos
        })

        if response.status_code == 200:
            return pd.read_csv(BytesIO(response.content))
        else:
            # Fallback to LFS media URL
            lfs_url = "https://media.githubusercontent.com/media/kheiriddine/Models_Repositories/main/gradiant_poc-code_base-02111a06fe82/gradiant_poc-code_base-02111a06fe82/0.%20Donnees%20Mock-up/Pharma_Procurement_Transactions.csv"
            lfs_response = requests.get(lfs_url)
            return pd.read_csv(BytesIO(lfs_response.content))

    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None



@st.cache_resource
def load_xgb_model():
    """Load XGBoost model and components using direct URLs"""
    try:
        # Direct URLs for the model and label encoder
        model_url = "https://raw.githubusercontent.com/kheiriddine/Models_Repositories/main/model_fin_xgboost_classifier/xgboost_model.json"
        label_encoder_url = "https://raw.githubusercontent.com/kheiriddine/Models_Repositories/main/model_fin_xgboost_classifier/label_encoder.json"

        # Load model
        response = requests.get(model_url)
        if response.status_code != 200:
            st.error(f"Failed to load model from {model_url}")
            return None, None, None

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        model = XGBClassifier()
        model.load_model(tmp_path)
        os.unlink(tmp_path)

        # Load label encoder
        le_response = requests.get(label_encoder_url)
        if le_response.status_code != 200:
            st.error(f"Failed to load label encoder from {label_encoder_url}")
            return None, None, None

        le_data = le_response.json()
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(le_data['classes'])

        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        return model, embedder, label_encoder

    except Exception as e:
        st.error(f"XGBoost loading error: {str(e)}")
        return None, None, None

@st.cache_resource
def load_bert_model():
    """Load BERT model and components using direct URLs and safetensors format"""
    try:
        # Direct URLs for the BERT model files
        base_url = "https://raw.githubusercontent.com/kheiriddine/Models_Repositories/main//bert_model"
        base_url_media = "https://media.githubusercontent.com/media/kheiriddine/Models_Repositories/main//bert_model"
        files = {
            'config.json': f"{base_url}/config.json",
            'model.safetensors': f"{base_url_media}/model.safetensors",  # Use the safetensors file
            'vocab.txt': f"{base_url}/vocab.txt",
            'special_tokens_map.json': f"{base_url}/special_tokens_map.json",
            'tokenizer_config.json': f"{base_url}/tokenizer_config.json"
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Download each file and save to the temporary directory
            for file_name, file_url in files.items():
                response = requests.get(file_url)
                if response.status_code != 200:
                    st.error(f"Failed to load {file_name} from {file_url}")
                    return None, None, None
                with open(os.path.join(temp_dir, file_name), 'wb') as f:
                    f.write(response.content)

            # Load label encoder using the same path as XGBoost
            label_encoder_url = "https://raw.githubusercontent.com/kheiriddine/Models_Repositories/main/model_fin_xgboost_classifier/label_encoder.json"
            le_response = requests.get(label_encoder_url)
            if le_response.status_code != 200:
                st.error(f"Failed to load label encoder from {label_encoder_url}")
                return None, None, None

            le_data = le_response.json()
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(le_data['classes'])

            # Load components
            model = BertForSequenceClassification.from_pretrained(temp_dir)
            tokenizer = BertTokenizer.from_pretrained(temp_dir)

            return model, tokenizer, label_encoder

    except Exception as e:
        st.error(f"BERT loading error: {str(e)}")
        return None, None, None
@st.cache_resource
def load_distilbert_model():
    """Load DistilBERT model and components using direct URLs"""
    try:
        # Direct URLs for the DistilBERT model files
        base_url = "https://raw.githubusercontent.com/kheiriddine/Models_Repositories/main/DistilBert_model"
        base_url_media = "https://media.githubusercontent.com/media/kheiriddine/Models_Repositories/main//DistilBert_model"
        files = {
            'config.json': f"{base_url}/config.json",
            'model.safetensors': f"{base_url_media}/model.safetensors",
            'vocab.txt': f"{base_url}/vocab.txt",
            'special_tokens_map.json': f"{base_url}/special_tokens_map.json",
            'tokenizer_config.json': f"{base_url}/tokenizer_config.json"
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Download each file and save to the temporary directory
            for file_name, file_url in files.items():
                response = requests.get(file_url)
                if response.status_code != 200:
                    st.error(f"Failed to load {file_name} from {file_url}")
                    return None, None, None
                with open(os.path.join(temp_dir, file_name), 'wb') as f:
                    f.write(response.content)

            # Load label encoder using the same path as XGBoost
            label_encoder_url = "https://raw.githubusercontent.com/kheiriddine/Models_Repositories/main/model_fin_xgboost_classifier/label_encoder.json"
            le_response = requests.get(label_encoder_url)
            if le_response.status_code != 200:
                st.error(f"Failed to load label encoder from {label_encoder_url}")
                return None, None, None

            le_data = le_response.json()
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(le_data['classes'])

            # Load components
            model = DistilBertForSequenceClassification.from_pretrained(temp_dir)
            tokenizer = DistilBertTokenizer.from_pretrained(temp_dir)

            return model, tokenizer, label_encoder

    except Exception as e:
        st.error(f"DistilBERT loading error: {str(e)}")
        return None, None, None


# Main Application Flow
def main():
    # Initialize session state
    if 'classification_result' not in st.session_state:
        st.session_state.classification_result = None

    # Load data
    df = load_procurement_data()

    if df is not None:
        st.subheader("Processed Data Preview")
        st.dataframe(df.head())

        # Supplier Analysis by Total Amount
        st.subheader("Supplier Analysis (by Amount)")
        if 'Supplier Name' in df.columns and 'Amount' in df.columns:
            supplier_amounts = df.groupby('Supplier Name')['Amount'].sum().reset_index()
            supplier_amounts['Percentage'] = (supplier_amounts['Amount'] / supplier_amounts['Amount'].sum()) * 100

            if 'Supplier_Category' in df.columns:
                supplier_amounts = df.groupby(['Supplier_Category', 'Supplier Name'])['Amount'].sum().reset_index()
                supplier_amounts['Percentage'] = supplier_amounts.groupby('Supplier_Category')['Amount'].apply(
                    lambda x: 100 * x / x.sum()
                )
                fig = px.bar(supplier_amounts.sort_values(['Supplier_Category', 'Amount'], ascending=[True, False]),
                             x='Supplier Name', y='Amount', color='Supplier_Category',
                             labels={'Amount': 'Total Amount'}, hover_data=['Percentage'], height=500)
                fig.update_layout(xaxis={'categoryorder': 'total descending'})
            else:
                fig = px.bar(supplier_amounts.sort_values('Amount', ascending=False).head(20),
                             x='Supplier Name', y='Amount', title='Top Suppliers by Spending (Amount)',
                             labels={'Amount': 'Total Amount'}, hover_data=['Percentage'], height=500)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Required columns (Supplier/Amount) not found")

        # Category Breakdown
        st.subheader("Category Breakdown")
        if 'Category' in df.columns and 'Amount' in df.columns:
            if 'Subcategory' in df.columns:
                cat_amounts = df.groupby(['Category', 'Subcategory'], as_index=False)['Amount'].sum()
                cat_totals = df.groupby('Category', as_index=False)['Amount'].sum()
                cat_amounts = cat_amounts.merge(cat_totals, on='Category', suffixes=('', '_total'))
                cat_amounts['Percentage'] = (cat_amounts['Amount'] / cat_amounts['Amount_total']) * 100
                fig = px.sunburst(cat_amounts, path=['Category', 'Subcategory'], values='Amount',
                                  title='Category/Subcategory', hover_data=['Percentage'], height=600)
            else:
                cat_amounts = df.groupby('Category', as_index=False)['Amount'].sum()
                total_amount = cat_amounts['Amount'].sum()
                cat_amounts['Percentage'] = (cat_amounts['Amount'] / total_amount) * 100
                fig = px.pie(cat_amounts, values='Amount', names='Category',
                             title='Category/Subcategory', hover_data=['Percentage'], height=500)

            st.plotly_chart(fig, use_container_width=True)
        else:
            missing_cols = []
            if 'Category' not in df.columns:
                missing_cols.append("'Category'")
            if 'Amount' not in df.columns:
                missing_cols.append("'Amount'")
            st.warning(f"Required columns {', '.join(missing_cols)} not found in the data")

        # Uncategorized Transactions Analysis
        st.subheader("Uncategorized Transactions")
        if 'Category' in df.columns:
            uncategorized = df['Category'].isna() | (df['Category'] == '')
            num_uncategorized = uncategorized.sum()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Uncategorized Transactions", num_uncategorized)
            with col2:
                if len(df) > 0:
                    st.metric("Percentage of Total", f"{(num_uncategorized / len(df)) * 100:.1f}%")

            if num_uncategorized > 0:
                st.write("Sample of uncategorized transactions:")
                st.dataframe(df.loc[uncategorized, ['Description', 'Amount']].head(10), height=200)
            else:
                st.success("All transactions are properly categorized!")
        else:
            st.warning("No 'Category' column found - cannot check categorization")

        # Hierarchical Spend Composition
        if 'Category' in df.columns and 'Amount' in df.columns:
            df_clean = df.copy()
            df_clean['Category'] = df_clean['Category'].fillna('Uncategorized')
            if 'Subcategory' in df_clean.columns:
                df_clean['Subcategory'] = df_clean['Subcategory'].fillna('Unspecified')
            if 'Supplier Name' in df_clean.columns:
                df_clean['Supplier Name'] = df_clean['Supplier Name'].fillna('Unknown Supplier')

            st.markdown("### Hierarchical Spend Composition")
            if 'Subcategory' in df_clean.columns and 'Supplier Name' in df_clean.columns:
                treemap_data = df_clean[df_clean['Amount'] > 0]
                fig_treemap = px.treemap(treemap_data, path=['Category', 'Subcategory', 'Supplier Name'],
                                         values='Amount', color='Amount', color_continuous_scale='Bluered',
                                         title='Spend Hierarchy', hover_data={'Amount': ':.2f'})
                fig_treemap.update_traces(
                    textinfo="label+value+percent parent",
                    texttemplate="<b>%{label}</b><br>%{value:$,.0f}<br>(%{percentParent:.1%})",
                    hovertemplate="<b>%{label}</b><br>Amount: %{value:$,.0f}<br>%{percentParent:.1%} of parent<extra></extra>"
                )
                st.plotly_chart(fig_treemap, use_container_width=True)
        else:
            st.warning("Required columns (Category/Amount) missing for analysis")

    st.title("Interactive Classification Simulator")
    st.subheader("Model Performance Comparison")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("XGBoost Accuracy", "74%", "Best for structured features")
    with col2:
        st.metric("BERT Accuracy", "76%", "Best for complex text")
    with col3:
        st.metric("DistilBERT Accuracy", "80.53%", "2x Faster and 60% smaller than BERT")
    


    # Model selection and classification
    #st.subheader("Select Classification Model")
    st.write("Choose a model to classify your transaction descriptions.")
    model_type = st.radio(
                "Model Type:",
                options=["XGBoost", "BERT", "DistilBERT"],
                horizontal=True,
                label_visibility="collapsed"
            )

    if model_type == "XGBoost":
        model, embedder, label_encoder = load_xgb_model()
    elif model_type == "BERT":
        model, tokenizer, label_encoder = load_bert_model()
    else:
        model, tokenizer, label_encoder = load_distilbert_model()

    # Text input for classification
    text_input = st.text_area("Enter text to classify:")

    # Classification button
    if st.button("Classify Text", type="primary"):
        if not text_input.strip():
            st.error("Please enter valid text to classify")
        else:
            try:
                with st.spinner("Classifying..."):
                    result = None
                    
                    if model_type == "XGBoost":
                        if model and embedder and label_encoder:
                            embedding = embedder.encode([text_input], show_progress_bar=False)
                            class_idx = model.predict(embedding)[0]
                            class_proba = model.predict_proba(embedding)[0]
                            class_name = label_encoder.inverse_transform([class_idx])[0]
                            prob = class_proba[class_idx]

                            result = {
                                'model': 'XGBoost',
                                'category': class_name,
                                'probability': prob * 100,
                                'all_probs': class_proba,
                                'classes': label_encoder.classes_
                            }
                        else:
                            st.error("XGBoost model not loaded properly")

                    elif model_type in ["BERT", "DistilBERT"]:
                        if model and tokenizer and label_encoder:
                            inputs = tokenizer(
                                text_input,
                                truncation=True,
                                padding=True,
                                max_length=512,
                                return_tensors="pt"
                            )

                            with torch.no_grad():
                                outputs = model(**inputs)
                                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                                pred_idx = torch.argmax(probs).item()
                                predicted_class = label_encoder.inverse_transform([pred_idx])[0]
                                prob = probs[0][pred_idx].item()

                            result = {
                                'model': model_type,
                                'category': predicted_class,
                                'probability': prob * 100,
                                'all_probs': probs[0].numpy(),
                                'classes': label_encoder.classes_
                            }
                        else:
                            st.error(f"{model_type} model not loaded properly")

                    if result is not None:
                        st.session_state.classification_result = result
                        st.success("Classification complete!")

            except Exception as e:
                st.error(f"Classification error: {str(e)}")

    # Display results
    if st.session_state.classification_result:
        result = st.session_state.classification_result
        
        st.subheader("Results")
        st.write(f"**Model Used:** {result['model']}")
        st.write(f"**Predicted Category:** {result['category']}")
        st.write(f"**Confidence:** {result['probability']:.2f}%")
        
        st.subheader("Probability Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Category**")
            for cls in result['classes']:
                st.write(cls)
        
        with col2:
            st.write("**Probability**")
            for prob in (result['all_probs'] * 100):
                st.write(f"{prob:.2f}%")

     # Model information
    with st.expander("ℹ️ Model Information"):
        if model_type == "XGBoost":
            st.markdown("""
        The XGBoost classifier demonstrated strong performance with an overall accuracy of **74%**. Key characteristics:
        - **Best in Marketing**: Achieved 0.89 precision and 0.86 F1-score
        - **Logistics Handling**: High recall (0.89) but lower precision (0.53)
        - **IT Challenges**: Lowest recall (0.53) indicating classification difficulties
        - **Consistent Performance**: Macro averages of 0.77 precision/0.74 recall
        
        The model shows particular strength in Manufacturing and Marketing categories, while benefiting from potential tuning in IT and Logistics classification.
        """)
    
        elif model_type == "BERT":
            st.markdown("""
        **Model**: `bert-base-uncased`  
        **Dataset**: 564 samples (fine-tuned)  
        **Training**:  
        - **Epochs**: 7  
        - **Dropout**: Applied (default: 0.1)  
        - **Accuracy**: **72.57%** (validation)  

        #### 📊 Key Observations:  
        - **Convergence**: Loss plateaued by epoch 7, suggesting sufficient training.  
        - **Generalization Gap**: Higher validation loss (0.712 vs. 0.565 training) indicates slight overfitting—expected with small datasets.  
        - **Performance**: Comparable to XGBoost (74% accuracy), but with stronger contextual understanding for complex text.  

        *Note: BERT outperforms XGBoost on nuanced text but requires more data for optimal results.*  
        """)
        elif model_type == "DistilBert":
            st.markdown("""
    	**Model**: A distilled version of `bert-base-uncased`, retaining most of its language understanding capabilities while being smaller 	and faster.
    	**Dataset**: Fine-tuned on 564 samples.
    	**Training**:
      		- **Epochs**: 7
     	 	- **Dropout**: Applied (default: 0.1)
        **Accuracy**: **80.53%** (validation)
    	**Key Observations**:
    	**Generalization**: The model shows strong generalization with an accuracy of 80.53%, demonstrating its ability to handle unseen 		data effectively.
    	**Performance**: DistilBERT achieves comparable performance to BERT with a smaller architecture, making it more efficient for real 	time applications.
    	**Efficiency**: The model's lighter architecture allows for faster inference times, making it suitable for environments with limited computational resources.
    	*Note: DistilBERT offers a good balance between performance and efficiency, making it an excellent choice for applications requiring 	quick and reliable transaction classification.*
        """) 
    st.title(" LLM as a judge")
    st.markdown("""
<style>
    .score-high { color: #2ecc71 !important; font-weight: bold; }
    .score-medium { color: #f39c12 !important; }
    .score-low { color: #e74c3c !important; }
</style>
""", unsafe_allow_html=True)

    #API Key Input
    api_key = st.text_input(
       " Enter your Together API Key:",
       type="password",
       help="Get your API key from https://together.ai"
)
    if not api_key:
        st.warning("Please enter your API key to continue")
        st.stop()

    try:
        client = Together(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize API client: {str(e)}")
        st.stop()


    def get_llm_response(description, category):
        prompt1=f"""
	    Rate from 0 to 100 how well the description fits the category.
	    Example:
	    - Description: 'Software development', Category: 'IT' → 95
	    - Description: 'Recruiting engineers', Category: 'HR & Training' → 80
	    Now rate: Description:repurpose distributed action-items from Berry , Category: R&D.
	    Return ONLY the number (0-100).
	    """
        prompt = f"""
	    Analyze how well this transaction description matches the category, then provide:
	    1. Detailed reasoning (marked with <think> tags)
	    2. Final score (0-100) on the last line
	    
	    Description: "{description}"
	    Category: "{category}"
	    
	    Follow this exact format:
	    <think>
	    [Your detailed reasoning here...]
	    </think>
	    [Final score as number]
	    """
	    
        try:
            response = client.chat.completions.create(
               model="deepseek-ai/DeepSeek-R1",
               messages=[{"role": "user", "content": prompt}],
               temperature=0.3,
               max_tokens=500
        )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"LLM Error: {str(e)}")
            return None

    def parse_response(response):
        if not response:
            return None, None
    
        # Extract reasoning
        reasoning_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
    
        # Extract score (last number in response)
        score_match = re.findall(r'\d+', response)
        score = int(score_match[-1]) if score_match else None
    
        if score is not None and (score < 0 or score > 100):
            st.error(f"Invalid score {score} - must be between 0-100")
            return reasoning, None
    
        return reasoning, score

    categories = ["IT", "Marketing", "R&D", "Manufacturing", "HR & Training", "Logistics"]

    col1, col2 = st.columns([3, 1])
    with col1:
        description = st.text_area(
        " Transaction Description:",
        value="Online purchase of computer equipment",
        height=100
    )
    with col2:
        category = st.selectbox(" Category:", categories, index=2)

    if st.button("Analyze", type="primary"):
        if not description.strip():
            st.error("Please enter a description")
        else:
            with st.spinner("Analyzing with DeepSeek-R1..."):
                start_time = time.time()
                response = get_llm_response(description, category)
                processing_time = time.time() - start_time
            
                if response:
                    reasoning, score = parse_response(response)
                
                    if score is not None:
                        st.divider()
                    
                        # Display reasoning
                        st.markdown("### 🤔 LLM Reasoning")
                        st.markdown(f'<div class="reasoning">{reasoning}</div>', unsafe_allow_html=True)
                    
                        # Display score
                        st.markdown("### 🎯 Final Score")
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Score", f"{score}/100")
                        with col2:
                            if score >= 80:
                                st.markdown('<p class="score-high">Excellent match</p>', unsafe_allow_html=True)
                            elif score >= 50:
                                st.markdown('<p class="score-medium">Moderate match</p>', unsafe_allow_html=True)
                            else:
                                st.markdown('<p class="score-low">Weak match</p>', unsafe_allow_html=True)
                    
                        # Visual indicator
                        st.progress(score/100, text=f"Confidence: {score}%")
                    
                        # Technical details
                        with st.expander("📊 Technical Details"):
                            st.write(f"**Processing Time:** {processing_time:.2f} seconds")
                            st.write(f"**Model:** DeepSeek-R1")
                            st.write(f"**Category:** {category}")


if __name__ == "__main__":
    main()
