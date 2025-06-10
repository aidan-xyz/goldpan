
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
from datetime import datetime, timedelta
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def deduplicate_memberships(df):
    """
    Deduplicate customer memberships based on renewal logic.
    
    Args:
        df: DataFrame with customer membership data
        
    Returns:
        DataFrame with deduplicated memberships
    """
    # Ensure we have the required columns
    required_columns = ['Customer Name', 'Created Date', 'Expiration Date']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the data")
    
    # Convert date columns to datetime
    df['Created Date'] = pd.to_datetime(df['Created Date'])
    df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])
    
    # Sort by Customer Name and Created Date (newest first)
    df_sorted = df.sort_values(['Customer Name', 'Created Date'], ascending=[True, False])
    
    # Group by Customer Name
    grouped = df_sorted.groupby('Customer Name')
    
    deduplicated_records = []
    
    for customer_name, group in grouped:
        group_list = group.to_dict('records')
        
        if len(group_list) == 1:
            # Single record for this customer, keep it
            deduplicated_records.extend(group_list)
            continue
        
        # Process multiple records for the same customer
        keep_records = []
        i = 0
        
        while i < len(group_list):
            current_record = group_list[i]
            
            # Check if this record should be kept
            should_keep = True
            
            # Look for renewals (records created within 7 days after expiration)
            for j in range(i + 1, len(group_list)):
                next_record = group_list[j]
                
                # Calculate days between expiration and next creation
                days_diff = (current_record['Expiration Date'] - next_record['Created Date']).days
                
                # If next record is created within 7 days after current expiration, it's a renewal
                if -7 <= days_diff <= 7:
                    should_keep = False
                    break
            
            if should_keep:
                keep_records.append(current_record)
            
            i += 1
        
        # If no records are kept (all were renewals), keep the most recent one
        if not keep_records:
            keep_records.append(group_list[0])
        
        deduplicated_records.extend(keep_records)
    
    # Convert back to DataFrame and sort by Customer Name
    result_df = pd.DataFrame(deduplicated_records)
    result_df = result_df.sort_values('Customer Name')
    
    return result_df

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Read the Excel file
            df = pd.read_excel(file)
            
            # Show preview of original data
            original_count = len(df)
            
            # Perform deduplication
            deduplicated_df = deduplicate_memberships(df)
            deduplicated_count = len(deduplicated_df)
            
            # Save the processed file to a temporary location
            temp_dir = tempfile.gettempdir()
            output_filename = f"deduplicated_{secure_filename(file.filename)}"
            output_path = os.path.join(temp_dir, output_filename)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                deduplicated_df.to_excel(writer, index=False, sheet_name='Deduplicated')
            
            return render_template('results.html', 
                                 original_count=original_count,
                                 deduplicated_count=deduplicated_count,
                                 reduction=original_count - deduplicated_count,
                                 filename=output_filename,
                                 preview_data=deduplicated_df.head(10).to_html(classes='table table-striped', index=False))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('upload_file'))
    else:
        flash('Invalid file type. Please upload an Excel file (.xlsx or .xls)')
        return redirect(url_for('upload_file'))

@app.route('/download/<filename>')
def download_file(filename):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    else:
        flash('File not found')
        return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
