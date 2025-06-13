from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
from datetime import datetime, timedelta
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
app.secret_key = 'your-secret-key-here' # REMINDER: Change this to a strong, securely stored key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_orders_and_add_value_tiers(membership_df, orders_df):
    """
    Process orders file and add Total Order Value and Estimated Savings columns to membership data.

    Args:
        membership_df: DataFrame with membership data (Cust ID)
        orders_df: DataFrame with orders data (SAP ID, Total Value)

    Returns:
        DataFrame with added columns
    """
    # Ensure we have the required columns
    if 'Cust ID' not in membership_df.columns:
        raise ValueError("Required column 'Cust ID' not found in membership data")
    if 'SAP ID' not in orders_df.columns:
        raise ValueError("Required column 'SAP ID' not found in orders data")
    if 'Total Value' not in orders_df.columns:
        raise ValueError("Required column 'Total Value' not found in orders data")

    # Convert Total Value to numeric, handling any non-numeric values
    orders_df['Total Value'] = pd.to_numeric(orders_df['Total Value'], errors='coerce').fillna(0)

    # Group orders by SAP ID and sum total values
    order_totals = orders_df.groupby('SAP ID')['Total Value'].sum().reset_index()
    order_totals.columns = ['SAP ID', 'Total Order Value']

    # Ensure both IDs are strings before merging
    membership_df['Cust ID'] = membership_df['Cust ID'].astype(str)
    order_totals['SAP ID'] = order_totals['SAP ID'].astype(str)

    # Merge membership data with order totals
    # Assuming Cust ID and SAP ID refer to the same customer identifier
    membership_df = membership_df.merge(
        order_totals, 
        left_on='Cust ID', 
        right_on='SAP ID', 
        how='left'
    )

    # Fill NaN values with 0 for customers with no orders
    membership_df['Total Order Value'] = membership_df['Total Order Value'].fillna(0)

    # Add Estimated Savings column (Total Order Value * 8%)
    membership_df['Estimated Savings'] = membership_df['Total Order Value'] * 0.08

    # Drop the SAP ID column as it's redundant with Cust ID
    if 'SAP ID' in membership_df.columns:
        membership_df = membership_df.drop('SAP ID', axis=1)

    return membership_df

def categorize_and_sort_memberships(df):
    """
    Categorize memberships by expiration status and sort them.

    Args:
        df: DataFrame with customer membership data

    Returns:
        DataFrame sorted by expiration category and date
    """
    # Ensure we have the required column
    if 'Expiration Date' not in df.columns:
        raise ValueError("Required column 'Expiration Date' not found in the data")

    # Ensure Expiration Date is datetime
    df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])

    # Get today's date
    today = pd.Timestamp.now().normalize()

    # Calculate days until expiration
    df['Days Until Expiration'] = (df['Expiration Date'] - today).dt.days

    # Categorize memberships
    def categorize_membership(days_until_exp):
        if days_until_exp < 0:
            return 'Expired'
        elif days_until_exp <= 30:
            return 'Expiring Soon'
        else:
            return 'Active'

    df['Expiration Category'] = df['Days Until Expiration'].apply(categorize_membership)

    # Define category order for sorting
    category_order = {'Expired': 0, 'Expiring Soon': 1, 'Active': 2}
    df['Category Order'] = df['Expiration Category'].map(category_order)

    # Sort by category first, then by expiration date ascending within each category
    df_sorted = df.sort_values(['Category Order', 'Expiration Date'], ascending=[True, True])

    # Remove helper columns
    df_sorted = df_sorted.drop(['Days Until Expiration', 'Category Order'], axis=1)

    return df_sorted

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

    # Sort by Customer Name and Created Date (oldest to newest)
    df_sorted = df.sort_values(['Customer Name', 'Created Date'], ascending=[True, True])

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

        for i in range(len(group_list)):
            current_record = group_list[i]
            should_keep = True

            # Check if the next record is a renewal within 7 days
            if i < len(group_list) - 1:
                next_record = group_list[i + 1]

                # Calculate days between current expiration and next creation
                days_diff = (next_record['Created Date'] - current_record['Expiration Date']).days

                # If next record starts within 7 days after current expires, it's a renewal
                if days_diff <= 7:
                    should_keep = False  # Remove the older record

            if should_keep:
                keep_records.append(current_record)

        deduplicated_records.extend(keep_records)

    # Convert back to DataFrame and sort by Customer Name
    result_df = pd.DataFrame(deduplicated_records)
    result_df = result_df.sort_values('Customer Name')

    return result_df

def format_for_hubspot_export(df):
    """
    Format the dataframe for HubSpot export by creating a single multi-select tag column,
    including 'Customer Name', 'Contact Email', 'Total Order Value', 'Estimated Savings',
    and removing specified columns.

    Args:
        df: DataFrame with processed membership data.
            Assumes 'Created Date' is the Membership Renewal Date.
            Assumes 'Customer Name' and 'Contact Email' are present in the input df.

    Returns:
        DataFrame formatted for HubSpot with a new multi-select column.
    """
    hubspot_df = df.copy()

    # Ensure date columns are in datetime format for calculations
    # Use errors='coerce' to turn unparseable dates into NaT
    if 'Created Date' in hubspot_df.columns:
        hubspot_df['Created Date'] = pd.to_datetime(hubspot_df['Created Date'], errors='coerce')
        hubspot_df = hubspot_df.rename(columns={'Created Date': 'Start Date'})

    if 'Expiration Date' in hubspot_df.columns:
        hubspot_df['Expiration Date'] = pd.to_datetime(hubspot_df['Expiration Date'], errors='coerce')

    # Convert to datetime.date objects for precise comparison (removes time component)
    # Use .mask to handle NaT values after .dt.date
    hubspot_df['Expiration Date_dt'] = hubspot_df['Expiration Date'].dt.date.mask(pd.isna(hubspot_df['Expiration Date']))
    hubspot_df['Start Date_dt'] = hubspot_df['Start Date'].dt.date.mask(pd.isna(hubspot_df['Start Date']))

    # Get today's date (date only)
    today = datetime.now().date()

    # Ensure 'Total Order Value' is numeric and handle potential missingness by defaulting to 0
    total_value_series = pd.to_numeric(hubspot_df.get('Total Order Value', pd.Series(0, index=hubspot_df.index)), errors='coerce').fillna(0)

    # Create a list column to hold tags for each row
    hubspot_df['Membership Status Tags List'] = [[] for _ in range(len(hubspot_df))]

    # --- Apply Tagging Logic and populate the list ---

    for idx, row in hubspot_df.iterrows():
        current_tags = []

        # High-Value Customer
        if total_value_series.iloc[idx] > 1000:
            current_tags.append('High-Value Customer')
        else:
            current_tags.append('Not High-Value Customer') # Explicitly add false state

        start_date = row['Start Date_dt']
        expiration_date = row['Expiration Date_dt']

        # Original Purchase Method (HFO Buy / Not HFO Buy)
        # Assuming 'Original Pu' is the column name for Original Purchase Method
        original_purchase_method = row.get('Original Pu') 
        if pd.notna(original_purchase_method) and str(original_purchase_method).strip() != '':
            current_tags.append('HFO Buy')
        else:
            current_tags.append('Not HFO Buy')

        # Ensure key dates are valid for date-based calculations
        if pd.notna(start_date) and pd.notna(expiration_date):
            # Active Membership
            if (today >= start_date) and (today < expiration_date):
                current_tags.append('Active Membership')
            else:
                current_tags.append('Not Active Membership') # Explicitly add false state

            # Days until expiration (handle NaT)
            days_until_expiration_val = (expiration_date - today).days if pd.notna(expiration_date) else None

            # Expiring Soon
            if (days_until_expiration_val is not None) and (0 <= days_until_expiration_val <= 30):
                current_tags.append('Expiring Soon')
            else:
                current_tags.append('Not Expiring Soon') # Explicitly add false state

            # Days since start (handle NaT)
            days_since_start_val = (today - start_date).days if pd.notna(start_date) else None

            # Recently Renewed
            if (days_since_start_val is not None) and (0 <= days_since_start_val <= 30):
                current_tags.append('Recently Renewed')
            else:
                current_tags.append('Not Recently Renewed') # Explicitly add false state
        else:
            # If dates are missing (NaT), explicitly tag as 'Not Active', 'Not Expiring Soon', 'Not Recently Renewed'
            current_tags.append('Not Active Membership')
            current_tags.append('Not Expiring Soon')
            current_tags.append('Not Recently Renewed')

        hubspot_df.at[idx, 'Membership Status Tags List'] = current_tags

    # Join the lists into a semicolon-separated string for the final column
    hubspot_df['Membership Status Tags'] = hubspot_df['Membership Status Tags List'].apply(lambda x: ';'.join(x))

    # --- Clean up temporary columns ---

    # Drop the temporary datetime objects used for calculations and the intermediate list column
    hubspot_df = hubspot_df.drop(['Expiration Date_dt', 'Start Date_dt', 'Membership Status Tags List'], axis=1)

    # Drop the 'Expiration Category' if it exists, as new multi-select column is desired
    if 'Expiration Category' in hubspot_df.columns:
        hubspot_df = hubspot_df.drop('Expiration Category', axis=1)

    # Convert the actual 'Start Date' and 'Expiration Date' columns to the desired string format *here*
    # This ensures they are in string format for the final CSV export, after all calculations.
    # Use fillna('') to replace any NaT values with empty strings in the output
    if 'Start Date' in hubspot_df.columns:
        hubspot_df['Start Date'] = pd.to_datetime(hubspot_df['Start Date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('')
    if 'Expiration Date' in hubspot_df.columns:
        hubspot_df['Expiration Date'] = pd.to_datetime(hubspot_df['Expiration Date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('')

    # --- Columns to remove from HubSpot output ---
    columns_to_remove_from_hubspot_output = [
        'Cust ID',      # Original Customer ID
        'City',
        'State/Regi',   # State/Region
        'Membersh',     # Original Membership field
        'Sales Doc',
        'Original Pu',  # Original Purchase (now part of tags, so remove column)
        'HFO',
        'Contact ID',
        'Contact Ni',   # Contact Nickname
        'Contact Pr',   # Contact Property/Phone
        # Also remove 'Start Date' and 'Expiration Date' if they are not desired as separate columns,
        # but you specified keeping them indirectly through the final desired output.
        # Keeping Start Date and Expiration Date as per your new desired list.
    ]

    # Drop columns if they exist in the DataFrame
    for col in columns_to_remove_from_hubspot_output:
        if col in hubspot_df.columns:
            hubspot_df = hubspot_df.drop(col, axis=1)

    # Define the exact final columns and their order for HubSpot output
    final_hubspot_columns = [
        'Customer Name',
        'Contact Email',
        'Membership Status Tags',
        'Total Order Value',
        'Estimated Savings'
    ]

    # Filter to only keep the final desired columns and ensure their order
    # This will also drop any other columns not explicitly listed here.
    hubspot_df = hubspot_df[[col for col in final_hubspot_columns if col in hubspot_df.columns]]

    return hubspot_df


@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process_file():
    # Check if both files are present
    if 'membership_file' not in request.files or 'orders_file' not in request.files:
        flash('Both membership and orders files are required')
        return redirect(request.url)

    membership_file = request.files['membership_file']
    orders_file = request.files['orders_file']

    if membership_file.filename == '' or orders_file.filename == '':
        flash('Both files must be selected')
        return redirect(request.url)

    # Get export format
    export_format = request.form.get('export_format', 'standard')

    if (membership_file and allowed_file(membership_file.filename) and 
        orders_file and allowed_file(orders_file.filename)):
        try:
            # Read both Excel files
            membership_df = pd.read_excel(membership_file)
            orders_df = pd.read_excel(orders_file)

            # Show preview of original data
            original_count = len(membership_df)

            # Perform deduplication on membership data
            deduplicated_df = deduplicate_memberships(membership_df)
            deduplicated_count = len(deduplicated_df)

            # Process orders and add value tiers
            processed_df = process_orders_and_add_value_tiers(deduplicated_df, orders_df)

            # Categorize and sort by expiration status (this adds 'Expiration Category')
            final_df = categorize_and_sort_memberships(processed_df)

            # Apply HubSpot formatting if selected
            if export_format == 'hubspot':
                final_df = format_for_hubspot_export(final_df)
                # Remove empty columns again after HubSpot formatting
                final_df = final_df.dropna(axis=1, how='all')
                file_extension = 'csv'
                output_filename = f"hubspot_{secure_filename(membership_file.filename).rsplit('.', 1)[0]}.csv"
            else:
                file_extension = 'xlsx'
                output_filename = f"processed_{secure_filename(membership_file.filename)}"

            # Save the processed file to a temporary location
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, output_filename)

            if file_extension == 'csv':
                final_df.to_csv(output_path, index=False)
            else:
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    final_df.to_excel(writer, index=False, sheet_name='Processed_Data')

            return render_template('results.html', 
                                   original_count=original_count,
                                   deduplicated_count=deduplicated_count,
                                   reduction=original_count - deduplicated_count,
                                   filename=output_filename,
                                   export_format=export_format,
                                   preview_data=final_df.head(10).to_html(classes='table table-striped', index=False))

        except Exception as e:
            flash(f'Error processing files: {str(e)}')
            return redirect(url_for('upload_file'))
    else:
        flash('Invalid file type. Please upload Excel files (.xlsx or .xls)')
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
