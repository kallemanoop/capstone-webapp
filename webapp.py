from distutils.log import debug
from fileinput import filename
import pandas as pd
from flask import *
import os
from werkzeug.utils import secure_filename
import retractometrics as ret
import numpy as np
import math

UPLOADED_FILES = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS= {'csv'}

app = Flask(__name__)

app.config['UPLOADED_FILES'] = UPLOADED_FILES

app.secret_key='Anoop@123'

@app.route('/',methods=['GET','POST'])
def uploadFile():
    if request.method == 'POST':
        f=request.files['file']
        data_filename=secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOADED_FILES'],data_filename))
        session['uploaded_data_file_path']=os.path.join(app.config['UPLOADED_FILES'],data_filename)
        return render_template('acknowledge.html')
    return render_template('index.html')

@app.route('/show_data')
def show_data():
    data_file_path = session.get('uploaded_data_file_path', None)

    if not data_file_path or not os.path.exists(data_file_path):
        return "<h1>No file uploaded or file missing.</h1>"

    print(f"Processing: {data_file_path}")

    required_columns = {'EID', 'Year', 'Cited by', 'Document Type', 'Funding Details'}
    
    try:
        df = pd.read_csv(data_file_path, encoding='unicode_escape')

        # Check if all required columns exist
        if not required_columns.issubset(df.columns):
            return f"<h1>Invalid file format. Required columns: {', '.join(required_columns)}</h1>"

        if df.empty:
            return "<h1>File is empty. Please upload a valid CSV.</h1>"

        # Convert 'Year' to numeric to prevent errors
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

        # Drop rows with NaN values in crucial columns
        df = df.dropna(subset=['Year', 'Cited by'])

        df.set_index('EID', drop=False, inplace=True, verify_integrity=True)

        total_citations = df['Cited by'].sum()
        total_papers = len(df)

        if total_papers == 0:
            return "<h1>No valid data found in the file.</h1>"

        data = {
            'Quantity of Publications': total_papers,
            'Total Citations': total_citations,
            'Average Citations per Paper (C/P)': total_citations / total_papers if total_papers else 0,
            'i10 Index': i10_index(df['Cited by']),
            'h index': h_index(df['Cited by']),
            'Mock_h index': ((total_citations ** 2) / total_papers) ** (1 / 3) if total_papers else 0,
            'e index': e_index(df['Cited by'], h_index(df['Cited by'])),
            'm index': h_index(df['Cited by']) / (2024 - df['Year'].min()) if not df['Year'].isna().all() else 0,
            'g index': g_index(df['Cited by']),
            's index': s_index(df['Cited by'], total_citations, total_papers),
            'Funding Details': df['Funding Details'].notna().sum()
        }

        yearly_citations = df.groupby('Year')['Cited by'].sum()
        yearly_h_indices = [h_index(df[df['Year'] == year]['Cited by']) for year in yearly_citations.index]
        data['t index'] = calculate_t_index(yearly_citations.tolist(), yearly_h_indices)

        doctype_counts = df['Document Type'].value_counts().to_dict()
        data.update(doctype_counts)

        df_html = pd.DataFrame.from_dict(data, orient='index').T.to_html(index=False)

        return render_template('show_data.html', data_table=df_html)

    except pd.errors.ParserError:
        return "<h1>Failed to read CSV. Make sure the file is properly formatted.</h1>"
    except Exception as e:
        return f"<h1>Unexpected error: {str(e)}</h1>"





def i10_index(citationSeries):
    """Calculate i10 index."""
    i10 = sum(1 for c in sorted(citationSeries, reverse=True) if c >= 10)
    return i10

def h_index(citationSeries):
    """Calculate h-index."""
    h = sum(1 for i, c in enumerate(sorted(citationSeries, reverse=True)) if c >= i + 1)
    return h

def e_index(citationSeries, hIndex):
    """Calculate e-index."""
    citations = sorted(citationSeries, reverse=True)
    totalCitations = sum(citations[:hIndex])
    return (totalCitations - hIndex**2) ** 0.5

def g_index(citationSeries):
    """Calculate g-index."""
    citations = sorted(citationSeries, reverse=True)
    g, totalCitations = 0, 0
    for c in citations:
        totalCitations += c
        if totalCitations >= (g + 1) ** 2:
            g += 1
        else:
            break
    return g

def s_index(citationSeries, totalCitations, totalPapers):
    """Calculate s-index."""
    if totalCitations == 0:
        return 0
    p_values = [c / totalCitations for c in citationSeries if c > 0]
    S = -sum(p * np.log(p) for p in p_values)
    S0 = math.log(totalPapers)
    return 0.25 * (totalCitations ** 0.5) * math.exp(S / S0)

def calculate_entropy(citations):
    """Calculate entropy."""
    total_citations = sum(citations)
    if total_citations == 0:
        return 0
    probabilities = [c / total_citations for c in citations if c > 0]
    return -sum(p * np.log(p) for p in probabilities)

def calculate_t_index(citations, h_indices):
    """Calculate t-index."""
    N = len(citations)
    if N == 0:
        return 0
    average_h_index = np.mean(h_indices)
    entropy_T = calculate_entropy(citations)
    normalization_factor = np.log(10 * N)
    consistency_u = np.exp(entropy_T / normalization_factor)
    return 4 * average_h_index * consistency_u




if __name__=='__main__':
    app.run(debug=True)
