import os
import io
import pandas as pd
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from groq import Groq

# --- Configuration & Initialization ---
load_dotenv()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Error initializing Groq client: {e}. Check GROQ_API_KEY.")
    groq_client = None

# --- Helper Function for Structured EDA ---
def generate_structured_eda(df: pd.DataFrame):
    """Generates a structured EDA report from a DataFrame."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    # Safely get memory usage
    memory_usage_line = [line for line in info_str.split('\n') if "memory usage" in line]
    memory_usage = memory_usage_line[0].split(': ')[1] if memory_usage_line else "N/A"

    eda_data = {
        'summary': {
            'rows': len(df),
            'columns': len(df.columns),
            'duplicate_rows': int(df.duplicated().sum()),
            'total_missing': int(df.isnull().sum().sum()),
            'memory_usage': memory_usage
        },
        'columns': [],
        'correlation_matrix_html': None
    }
    
    for col in df.columns:
        col_data = {}
        missing_count = int(df[col].isnull().sum())
        col_data['name'] = col
        col_data['missing_percent'] = round((missing_count / len(df)) * 100, 2)
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = 'Numeric'
            stats = df[col].describe()
            col_data['stats'] = {
                'mean': f"{stats.get('mean', 0):.2f}",
                'std': f"{stats.get('std', 0):.2f}",
                'min': f"{stats.get('min', 0):.2f}",
                'max': f"{stats.get('max', 0):.2f}",
            }
        else:
            col_type = 'Categorical'
            stats = df[col].describe()
            col_data['stats'] = {
                'unique_values': int(stats.get('unique', 0)),
                'top_value': str(stats.get('top', 'N/A')),
            }
        col_data['type'] = col_type
        eda_data['columns'].append(col_data)

    numeric_cols = df.select_dtypes(include=['number'])
    if not numeric_cols.empty and len(numeric_cols.columns) > 1:
        corr_matrix = numeric_cols.corr().round(2)
        eda_data['correlation_matrix_html'] = corr_matrix.to_html(
            classes='table table-sm table-bordered table-hover text-center', border=0
        )
    return eda_data

# --- API Routes ---
@app.route('/')
def home():
    """Serves the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles file upload, performs EDA, and returns all data as JSON."""
    if 'dataset' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    file = request.files['dataset']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)

        eda_report = generate_structured_eda(df)
        table_preview_html = df.head().to_html(classes='table table-striped table-hover', index=False)
        
        return jsonify({
            "filename": filename,
            "eda": eda_report,
            "preview_html": table_preview_html,
            "filepath": filepath
        })
    except Exception as e:
        return jsonify({"error": f"An error occurred while processing '{filename}': {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat questions using Groq."""
    data = request.get_json()
    question = data.get('question')
    filepath = data.get('filepath')

    if not all([question, filepath, groq_client]):
        return jsonify({"error": "Missing data or Groq client not initialized."}), 400

    try:
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        
        # Create a text summary for Groq's context
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        context = f"Dataset Summary:\nColumns and dtypes:\n{info_str}\nFirst 5 rows:\n{df.head().to_string()}"

        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a data analyst. Answer the user's question based on the provided data context. Be concise and clear."},
                {"role": "user", "content": prompt},
            ],
            model="llama3-8b-8192",
        )
        answer = chat_completion.choices[0].message.content
        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Render sets this
    app.run(host='0.0.0.0', port=port)        # Must bind to 0.0.0.0
