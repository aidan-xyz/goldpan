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
    This function will continue to add 'Expiration Category' for internal use,
    but it will be removed or superseded by other flags in final outputs.

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
    Deduplicate customer memberships by keeping the most recent (latest Expiration Date, then latest Created Date)
    record for each unique customer. This handles renewal scenarios where an older 'Expired' record
    might exist alongside a newer 'Active' record.

    Args:
        df: DataFrame with customer membership data. Must contain 'Customer Name', 'Created Date', 'Expiration Date'.

    Returns:
        DataFrame with deduplicated memberships.
    """
    # Ensure we have the required columns
    required_columns = ['Customer Name', 'Created Date', 'Expiration Date']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the data for deduplication.")

    # Convert date columns to datetime, coercing errors to NaT
    df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')
    df['Expiration Date'] = pd.to_datetime(df['Expiration Date'], errors='coerce')

    # Drop rows where essential columns for deduplication are NaT after conversion
    df.dropna(subset=['Customer Name', 'Created Date', 'Expiration Date'], inplace=True)

    # Sort by Customer Name, then by Expiration Date (descending - latest first),
    # then by Created Date (descending - most recent first as a tie-breaker).
    # This ensures that for each customer, the record with the furthest expiration
    # date is prioritized, and among those, the one with the most recent creation date.
    df_sorted = df.sort_values(
        by=['Customer Name', 'Expiration Date', 'Created Date'],
        ascending=[True, False, False]
    )

    # Drop duplicates, keeping the first occurrence (which is now the most recent/relevant record due to sorting)
    # The subset should be the unique identifier for a customer.
    deduplicated_df = df_sorted.drop_duplicates(subset=['Customer Name'], keep='first')

    # Reset index for cleaner output
    deduplicated_df = deduplicated_df.reset_index(drop=True)

    return deduplicated_df


def _calculate_all_membership_flags(df):
    """
    Helper function to calculate all boolean flags for membership status and value.
    Returns a DataFrame with original columns + new boolean flag columns (prefixed with _is_).
    """
    flagged_df = df.copy()

    # Ensure date columns are in datetime64[ns] format for consistent calculations
    # Use errors='coerce' to turn unparseable dates into NaT
    for col in ['Start Date', 'Expiration Date']:
        if col in flagged_df.columns:
            flagged_df[f'{col}_dt_ts'] = pd.to_datetime(flagged_df[col], errors='coerce')
        else:
            flagged_df[f'{col}_dt_ts'] = pd.Series([pd.NaT] * len(flagged_df), index=flagged_df.index)

    # Use pandas Timestamp for 'today' for consistent comparisons with datetime64[ns] series
    today_ts = pd.Timestamp.now().normalize()

    # Ensure 'Total Order Value' is numeric and handle potential missingness by defaulting to 0
    total_value_series = pd.to_numeric(flagged_df.get('Total Order Value', pd.Series(0, index=flagged_df.index)), errors='coerce').fillna(0)

    # --- Calculate Flags ---

    # High-Value Customer
    flagged_df['_is_high_value'] = total_value_series > 1000

    # Original Purchase Method (HFO Buy)
    # Check if 'Original Pu' column exists and has non-empty/non-NaN values
    if 'Original Pu' in flagged_df.columns:
        flagged_df['_is_hfo_buy'] = flagged_df['Original Pu'].apply(lambda x: pd.notna(x) and str(x).strip() != '')
    else:
        flagged_df['_is_hfo_buy'] = False # Default to False if column doesn't exist

    # Active Membership
    # Condition: Today >= Start Date AND Today < Expiration Date
    flagged_df['_is_active_membership'] = (today_ts >= flagged_df['Start Date_dt_ts']) & (today_ts < flagged_df['Expiration Date_dt_ts'])
    flagged_df['_is_active_membership'] = flagged_df['_is_active_membership'].fillna(False) # Handle NaT from date comparisons

    # Days until expiration calculation (difference between Timestamps yields Timedelta)
    days_until_expiration = (flagged_df['Expiration Date_dt_ts'] - today_ts).dt.days

    # Expiring Soon
    # Condition: Days until expiration is between 0 and 30 (inclusive)
    flagged_df['_is_expiring_soon'] = (days_until_expiration >= 0) & (days_until_expiration <= 30)
    flagged_df['_is_expiring_soon'] = flagged_df['_is_expiring_soon'].fillna(False) # Handle NaT

    # Days since start calculation
    days_since_start = (today_ts - flagged_df['Start Date_dt_ts']).dt.days

    # Recently Renewed
    # Condition: Days since start is between 0 and 30 (inclusive)
    flagged_df['_is_recently_renewed'] = (days_since_start >= 0) & (days_since_start <= 30)
    flagged_df['_is_recently_renewed'] = flagged_df['_is_recently_renewed'].fillna(False) # Handle NaT

    return flagged_df

def add_individual_boolean_tags_for_excel(df):
    """
    Adds individual boolean columns for each tag status for Excel export.
    Outputs as True/False booleans (actual Python booleans, which pandas will write as TRUE/FALSE).
    """
    excel_df = df.copy()

    # Calculate all core boolean flags first
    excel_df = _calculate_all_membership_flags(excel_df)

    # --- Populate new boolean columns based on calculated flags ---

    excel_df['High-Value Customer (True)'] = excel_df['_is_high_value']
    excel_df['High-Value Customer (False)'] = ~excel_df['_is_high_value']

    excel_df['HFO Buy (True)'] = excel_df['_is_hfo_buy']
    excel_df['HFO Buy (False)'] = ~excel_df['_is_hfo_buy']

    excel_df['Active Membership (True)'] = excel_df['_is_active_membership']
    excel_df['Active Membership (False)'] = ~excel_df['_is_active_membership']

    excel_df['Expiring Soon (True)'] = excel_df['_is_expiring_soon']
    excel_df['Expiring Soon (False)'] = ~excel_df['_is_expiring_soon']

    excel_df['Recently Renewed (True)'] = excel_df['_is_recently_renewed']
    excel_df['Recently Renewed (False)'] = ~excel_df['_is_recently_renewed']

    # Drop temporary datetime columns and the intermediate flag columns
    excel_df = excel_df.drop(columns=[
        'Start Date_dt_ts', 'Expiration Date_dt_ts',
        '_is_high_value', '_is_hfo_buy', '_is_active_membership',
        '_is_expiring_soon', '_is_recently_renewed'
    ], errors='ignore')

    # Remove 'Expiration Category' if it exists, as it's replaced by detailed booleans
    if 'Expiration Category' in excel_df.columns:
        excel_df = excel_df.drop('Expiration Category', axis=1)

    return excel_df


def format_for_hubspot_export(df):
    """
    Formats the dataframe for HubSpot export by creating a single multi-select tag column,
    including 'Customer Name', 'Contact Email', 'Total Order Value', 'Estimated Savings',
    and removing specified columns.

    Args:
        df: DataFrame with processed membership data.

    Returns:
        DataFrame formatted for HubSpot with a new multi-select column.
    """
    hubspot_df = df.copy()

    # Format dates toimbangkan-MM-DD string format for HubSpot compatibility
    # Do this BEFORE calling _calculate_all_membership_flags if those dates are needed
    # for the helper function. The helper creates its own _dt_ts versions.
    if 'Created Date' in hubspot_df.columns:
        hubspot_df['Created Date'] = pd.to_datetime(hubspot_df['Created Date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('')
        hubspot_df = hubspot_df.rename(columns={'Created Date': 'Start Date'})

    if 'Expiration Date' in hubspot_df.columns:
        hubspot_df['Expiration Date'] = pd.to_datetime(hubspot_df['Expiration Date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('')

    # Calculate all core boolean flags first
    hubspot_df = _calculate_all_membership_flags(hubspot_df)

    # Create a list column to hold tags for each row
    hubspot_df['Membership Status Tags List'] = [[] for _ in range(len(hubspot_df))]

    # --- Populate the list based on calculated flags ---

    for idx, row in hubspot_df.iterrows():
        current_tags = []

        # High-Value Customer
        if row['_is_high_value']:
            current_tags.append('High-Value Customer')
        else:
            current_tags.append('Not High-Value Customer') 

        # Original Purchase Method (HFO Buy / Not HFO Buy)
        if row['_is_hfo_buy']:
            current_tags.append('HFO Buy')
        else:
            current_tags.append('Not HFO Buy')

        # Active Membership
        if row['_is_active_membership']:
            current_tags.append('Active Membership')
        else:
            current_tags.append('Not Active Membership') 

        # Expiring Soon
        if row['_is_expiring_soon']:
            current_tags.append('Expiring Soon')
        else:
            current_tags.append('Not Expiring Soon') 

        # Recently Renewed
        if row['_is_recently_renewed']:
            current_tags.append('Recently Renewed')
        else:
            current_tags.append('Not Recently Renewed') 

        hubspot_df.at[idx, 'Membership Status Tags List'] = current_tags

    # Join the lists into a semicolon-separated string for the final column
    hubspot_df['Membership Status Tags'] = hubspot_df['Membership Status Tags List'].apply(lambda x: ';'.join(x))

    # --- Clean up temporary columns ---

    # Drop the temporary datetime objects used for calculations and the intermediate list column
    hubspot_df = hubspot_df.drop(columns=[
        'Start Date_dt_ts', 'Expiration Date_dt_ts', # From _calculate_all_membership_flags
        'Membership Status Tags List', 
        '_is_high_value', '_is_hfo_buy', '_is_active_membership', # The boolean flags themselves
        '_is_expiring_soon', '_is_recently_renewed'
    ], errors='ignore')

    # Drop the 'Expiration Category' if it exists
    if 'Expiration Category' in hubspot_df.columns:
        hubspot_df = hubspot_df.drop('Expiration Category', axis=1)

    # --- Columns to remove from HubSpot output explicitly ---
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
        'Start Date',   # Explicitly removing if you only want the tags for dates
        'Expiration Date' # Explicitly removing if you only want the tags for dates
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
            # This step is kept as it might be used internally or for other non-HubSpot exports
            final_df = categorize_and_sort_memberships(processed_df)

            # Apply HubSpot formatting if selected
            if export_format == 'hubspot':
                final_df = format_for_hubspot_export(final_df)
                # Remove empty columns again after HubSpot formatting
                final_df = final_df.dropna(axis=1, how='all')
                file_extension = 'csv'
                output_filename = f"hubspot_{secure_filename(membership_file.filename).rsplit('.', 1)[0]}.csv"
            else: # Standard Excel output
                # Add individual boolean tag columns for Excel verification
                final_df = add_individual_boolean_tags_for_excel(final_df)
                # Remove any columns that are now redundant or not desired in the Excel output
                # (e.g., Expiration Category if it exists and is now redundant)
                final_df = final_df.dropna(axis=1, how='all') 
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
