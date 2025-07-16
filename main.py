from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
from datetime import datetime, timedelta
import os
from werkzeug.utils import secure_filename
import tempfile
import re # For email domain extraction

app = Flask(__name__)
app.secret_key = 'your-secret-key-here' # REMINDER: Change this to a strong, securely stored key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

# List of common free email domains to exclude from company-level deduplication
COMMON_FREE_EMAIL_DOMAINS = {
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com',
    'icloud.com', 'protonmail.com', 'mail.com', 'gmx.com', 'zoho.com',
    'live.com', 'msn.com', 'yandex.com', 'qq.com', '163.com', 'sina.com'
}

# Common acronyms to preserve in uppercase during title recasing
COMMON_ACRONYMS = {
    'LLC', 'INC', 'CO', 'CORP', 'LTD', 'PLC', 'GMBH', 'AG', 'SA', 'NV', 'BV', 
    'PTE', 'S.A.', 'C.A.', 'S.R.L.', 'SPA', 'SAS', 'SARL', 'KG', 'OHG', 'LP', 
    'LLP', 'PC', 'PA', 'PS', 'SC', 'SD', 'USA', 'UK', 'CA', 'DE', 'FR', 'IT', 
    'JP', 'MX', 'AU', 'NZ', 'EU', 'NA', 'APAC', 'EMEA', 'LATAM', 'WC'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def recase_company_name(name):
    """
    Recases a company name to title case, preserving common acronyms in uppercase.
    Handles various delimiters and ensures consistent spacing.
    """
    if not isinstance(name, str) or not name:
        return name

    # Convert to uppercase first, then split by non-alphanumeric characters or spaces
    # This helps in handling names like "ABC-DEF" or "GHI.JKL"
    words = re.split(r'[^a-zA-Z0-9]+', name.upper())

    recase_words = []
    for word in words:
        if word in COMMON_ACRONYMS:
            recase_words.append(word)
        elif word: # Ensure word is not empty
            recase_words.append(word.capitalize())

    # Initialize final_name here to ensure it always has a value
    final_name = ' '.join(recase_words)

    # Specific replacements for common acronyms that might have been title-cased if not in the list
    # This loop ensures that even if an acronym was part of a hyphenated word, it gets corrected.
    for acr in sorted(list(COMMON_ACRONYMS), key=len, reverse=True): # Sort by length to match longer acronyms first
        # Use regex to match whole words to avoid replacing parts of other words (e.g., 'INC' in 'INCORPORATED')
        final_name = re.sub(r'\b' + acr.capitalize() + r'\b', acr, final_name)

    return final_name.strip()


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

def _extract_domain_from_email(email):
    """Safely extracts the domain from an email address."""
    if pd.notna(email) and isinstance(email, str) and '@' in email:
        return email.split('@')[-1].lower()
    return None

def deduplicate_memberships(df):
    """
    Deduplicate customer memberships by prioritizing the most recent record (latest Expiration Date,
    then latest Created Date) for each unique customer. Deduplication logic is based on:
    1. Email domain for corporate emails (excluding common free domains).
    2. Full contact email for personal/free emails.

    Args:
        df: DataFrame with customer membership data. Must contain 'Customer Name', 'Contact Email',
            'Created Date', 'Expiration Date'.

    Returns:
        DataFrame with deduplicated memberships.
    """
    deduplicated_df_parts = []

    # Ensure required columns exist and are converted/cleaned
    required_cols = ['Customer Name', 'Contact Email', 'Created Date', 'Expiration Date']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the data for deduplication.")

    # Convert date columns to datetime, coercing errors to NaT
    df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')
    df['Expiration Date'] = pd.to_datetime(df['Expiration Date'], errors='coerce')

    # Create an internal normalized Customer Name for better grouping if needed (used for internal sorting in this function)
    df['Internal_Normalized_Customer_Name'] = df['Customer Name'].astype(str).str.upper().str.strip().str.replace(r'[^A-Z0-9\s]', '', regex=True).str.replace(r'\s+', ' ', regex=True)

    # Extract email domains for internal deduplication logic
    df['Internal_Email_Domain'] = df['Contact Email'].apply(_extract_domain_from_email)

    # Drop rows where essential columns for deduplication are NaT or None after conversion/extraction
    df_cleaned = df.dropna(subset=['Customer Name', 'Contact Email', 'Created Date', 'Expiration Date', 'Internal_Email_Domain']).copy()

    # Separate into corporate and personal email domains
    corporate_emails_df = df_cleaned[~df_cleaned['Internal_Email_Domain'].isin(COMMON_FREE_EMAIL_DOMAINS)].copy()
    personal_emails_df = df_cleaned[df_cleaned['Internal_Email_Domain'].isin(COMMON_FREE_EMAIL_DOMAINS)].copy()

    # --- Deduplicate Corporate Emails by Domain ---
    if not corporate_emails_df.empty:
        corporate_emails_df_sorted = corporate_emails_df.sort_values(
            by=['Internal_Email_Domain', 'Expiration Date', 'Created Date'],
            ascending=[True, False, False] # Latest Expiration Date, then Latest Created Date
        )
        deduplicated_df_parts.append(corporate_emails_df_sorted.drop_duplicates(subset=['Internal_Email_Domain'], keep='first'))

    # --- Deduplicate Personal Emails by Full Email Address ---
    if not personal_emails_df.empty: # Ensure personal_emails_df is not empty before processing
        personal_emails_df_sorted = personal_emails_df.sort_values(
            by=['Contact Email', 'Expiration Date', 'Created Date'], # Use full email as identifier
            ascending=[True, False, False] # Latest Expiration Date, then Latest Created Date
        )
        deduplicated_df_parts.append(personal_emails_df_sorted.drop_duplicates(subset=['Contact Email'], keep='first'))

    # Combine the deduplicated parts
    if deduplicated_df_parts:
        result_df = pd.concat(deduplicated_df_parts, ignore_index=True)
    else:
        result_df = pd.DataFrame(columns=df.columns) # Return empty DataFrame if no data

    # Drop temporary internal columns before returning
    result_df = result_df.drop(columns=['Internal_Email_Domain', 'Internal_Normalized_Customer_Name'], errors='ignore')

    # Sort the final combined DataFrame by Customer Name (original column)
    result_df = result_df.sort_values('Customer Name').reset_index(drop=True)

    return result_df


def _calculate_all_membership_flags(df):
    """
    Helper function to calculate all boolean flags for membership status and value.
    Returns a DataFrame with original columns + new boolean flag columns (prefixed with _is_).
    Also adds 'Email Domain' for output.
    """
    flagged_df = df.copy()

    # Ensure date columns are in datetime64[ns] format for consistent calculations
    # Use errors='coerce' to turn unparseable dates into NaT
    for col_name in ['Start Date', 'Expiration Date']:
        if col_name in flagged_df.columns:
            flagged_df[f'{col_name}_dt_ts'] = pd.to_datetime(flagged_df[col_name], errors='coerce')
        else:
            flagged_df[f'{col_name}_dt_ts'] = pd.Series([pd.NaT] * len(flagged_df), index=flagged_df.index)

    # Use pandas Timestamp for 'today' for consistent comparisons with datetime64[ns] series
    today_ts = pd.Timestamp.now().normalize()

    # Ensure 'Total Order Value' is numeric and handle potential missingness by defaulting to 0
    total_value_series = pd.to_numeric(flagged_df.get('Total Order Value', pd.Series(0, index=flagged_df.index)), errors='coerce').fillna(0)

    # --- Add Email Domain for output ---
    flagged_df['Email Domain'] = flagged_df['Contact Email'].apply(_extract_domain_from_email)


    # --- Calculate Flags ---

    # High-Value Customer
    flagged_df['_is_high_value'] = total_value_series > 1000

    # Original Purchase Method (HFO Buy)
    # Check if 'Original Purchase Method' column exists and has non-empty/non-NaN values
    if 'Original Purchase Method' in flagged_df.columns:
        flagged_df['_is_hfo_buy'] = flagged_df['Original Purchase Method'].apply(lambda x: pd.notna(x) and str(x).strip() != '')
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

def _generate_membership_status_tags_list(df_row):
    """
    Generates a list of membership status tags for a single row based on boolean flags.
    This consolidates logic used by both HubSpot and Excel output formatting.
    Assumes _is_high_value, _is_hfo_buy, etc., flags are already present in df_row.
    """
    current_tags = []

    # WC History - always true for membership data
    current_tags.append('WC History')

    # High-Value Customer
    if df_row['_is_high_value']:
        current_tags.append('High-Value Customer')
    else:
        current_tags.append('Not High-Value Customer') 

    # Original Purchase Method (HFO Buy / Not HFO Buy)
    if df_row['_is_hfo_buy']:
        current_tags.append('HFO Buy')
    else:
        current_tags.append('Not HFO Buy')

    # Active Membership
    if df_row['_is_active_membership']:
        current_tags.append('Active Membership')
    else:
        current_tags.append('Not Active Membership') 

    # Expiring Soon
    if df_row['_is_expiring_soon']:
        current_tags.append('Expiring Soon')
    else:
        current_tags.append('Not Expiring Soon') 

    # Recently Renewed
    if df_row['_is_recently_renewed']:
        current_tags.append('Recently Renewed')
    else:
        current_tags.append('Not Recently Renewed') 

    return current_tags


def add_individual_boolean_tags_for_excel(df):
    """
    Adds individual boolean columns for each tag status for Excel export.
    Outputs as True/False booleans (actual Python booleans, which pandas will write as TRUE/FALSE).
    Includes the single 'Email Domain' column.
    """
    excel_df = df.copy()

    # Calculate all core boolean flags and add Email Domain
    excel_df = _calculate_all_membership_flags(excel_df)

    # --- Populate new boolean columns based on calculated flags ---

    # WC History - always true for membership data
    excel_df['WC History (True)'] = True
    excel_df['WC History (False)'] = False

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

    # Drop temporary datetime columns, and the intermediate flag columns
    excel_df = excel_df.drop(columns=[
        'Start Date_dt_ts', 'Expiration Date_dt_ts',
        '_is_high_value', '_is_hfo_buy', '_is_active_membership',
        '_is_expiring_soon', '_is_recently_renewed'
    ], errors='ignore')

    # Remove 'Expiration Category' if it exists, as it's replaced by detailed booleans
    if 'Expiration Category' in excel_df.columns:
        excel_df = excel_df.drop('Expiration Category', axis=1)

    return excel_df


def _generate_prospect_tags_list(df_row):
    """
    Generates a list of prospect tags for a single row based on boolean flags.
    Similar to membership tags but for prospects with no WC history.
    """
    current_tags = []

    # WC History - always false for prospects
    current_tags.append('No WC History')

    # High-Value Customer
    if df_row['Is High-Value Prospect']:
        current_tags.append('High-Value Customer')
    else:
        current_tags.append('Not High-Value Customer') 

    return current_tags

def add_individual_boolean_tags_for_prospects_excel(df):
    """
    Adds individual boolean columns for each tag status for Excel export for prospects.
    Outputs as True/False booleans for prospects with no WC history.
    """
    excel_df = df.copy()

    # --- Populate new boolean columns based on prospect flags ---

    # WC History - always false for prospects
    excel_df['WC History (True)'] = False
    excel_df['WC History (False)'] = True

    excel_df['High-Value Customer (True)'] = excel_df['Is High-Value Prospect']
    excel_df['High-Value Customer (False)'] = ~excel_df['Is High-Value Prospect']

    return excel_df

def process_phase_two_orders(orders_df):
    """
    Process orders file to identify non-WC prospects (Phase 2).

    Args:
        orders_df: DataFrame with orders data

    Returns:
        tuple: (final_prospect_data, high_value_prospects)
    """
    # Ensure required columns exist
    required_cols = ['SAP ID', 'Customer Name', 'Total Value']
    for col in required_cols:
        if col not in orders_df.columns:
            raise ValueError(f"Required column '{col}' not found in the orders file.")

    # Ensure IDs and Names are strings and clean 'nan' values
    orders_df['SAP ID'] = orders_df['SAP ID'].astype(str).replace('nan', '').replace('<NA>', '')
    orders_df['Customer Name'] = orders_df['Customer Name'].astype(str).replace('nan', '').replace('<NA>', '')

    # Convert Total Value to numeric, handling any non-numeric values
    orders_df['Total Value'] = pd.to_numeric(orders_df['Total Value'], errors='coerce').fillna(0)

    # Ensure 'WC Order' column exists and standardize values
    if 'WC Order' not in orders_df.columns:
        orders_df['WC Order'] = ''  # If column doesn't exist, treat as no 'yes' entries
    orders_df['WC Order'] = orders_df['WC Order'].astype(str).str.lower().str.strip()

    # --- Step 1: Determine which SAP IDs ever had a 'yes' WC Order ---
    wc_status_per_sap_id = orders_df.groupby('SAP ID')['WC Order'].apply(lambda x: (x == 'yes').any()).reset_index()
    wc_status_per_sap_id.columns = ['SAP ID', '_has_any_wc_order_yes']

    # Filter to get only SAP IDs that NEVER had a 'yes' WC Order
    non_wc_order_sap_ids = wc_status_per_sap_id[~wc_status_per_sap_id['_has_any_wc_order_yes']]['SAP ID']

    # Filter the original orders_df to include only these non-WC order companies
    clean_orders_df = orders_df[orders_df['SAP ID'].isin(non_wc_order_sap_ids)].copy()

    # --- Step 2: Aggregate total orders for these clean companies ---
    final_prospect_data = clean_orders_df.groupby(['SAP ID', 'Customer Name']).agg(
        Total_Order_Value=('Total Value', 'sum')
    ).reset_index()

    final_prospect_data.rename(columns={'SAP ID': 'Cust ID'}, inplace=True)

    # --- Step 3: Calculate Potential WC Savings (Total Order Value * 0.05) ---
    final_prospect_data['Potential WC Savings'] = final_prospect_data['Total_Order_Value'] * 0.05

    # --- Step 4: Identify High-Value Prospects ---
    final_prospect_data['Is High-Value Prospect'] = final_prospect_data['Total_Order_Value'] > 1000

    # Apply smart title recasing to Customer Name for Phase 2
    final_prospect_data['Customer Name'] = final_prospect_data['Customer Name'].apply(recase_company_name)

    # --- Step 5: Add tagging system for prospects ---
    final_prospect_data['Prospect Status Tags List'] = final_prospect_data.apply(_generate_prospect_tags_list, axis=1)
    final_prospect_data['Prospect Status Tags'] = final_prospect_data['Prospect Status Tags List'].apply(lambda x: ';'.join(x))

    # Prepare high-value prospects data (also recase Customer Name here)
    high_value_prospects = final_prospect_data[final_prospect_data['Is High-Value Prospect']][['Customer Name', 'Cust ID', 'Total_Order_Value', 'Potential WC Savings', 'Prospect Status Tags']].copy()
    high_value_prospects['Customer Name'] = high_value_prospects['Customer Name'].apply(recase_company_name)

    # --- Final Column Renames and Ordering for Output ---
    final_prospect_data.rename(columns={'Total_Order_Value': 'Total Order Value'}, inplace=True)
    
    # Drop the temporary list column
    final_prospect_data = final_prospect_data.drop('Prospect Status Tags List', axis=1)
    
    output_columns = [
        'Customer Name',
        'Cust ID',
        'Total Order Value',
        'Potential WC Savings',
        'Is High-Value Prospect',
        'Prospect Status Tags'
    ]
    final_prospect_data = final_prospect_data[output_columns]

    return final_prospect_data, high_value_prospects


def format_for_hubspot_export(df):
    """
    Formats the dataframe for HubSpot export by creating a single multi-select tag column,
    including 'Customer Name', 'Contact Email', 'Cust ID', 'Email Domain',
    'Total Order Value', 'Estimated Savings', and removing specified columns.

    Args:
        df: DataFrame with processed membership data.

    Returns:
        DataFrame formatted for HubSpot with a new multi-select column.
    """
    hubspot_df = df.copy()

    # Format dates toYYYY-MM-DD string format for HubSpot compatibility
    # Do this BEFORE calling _calculate_all_membership_flags if those dates are needed
    # for the helper function. The helper creates its own _dt_ts versions.
    if 'Created Date' in hubspot_df.columns:
        hubspot_df['Created Date'] = pd.to_datetime(hubspot_df['Created Date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('')
        hubspot_df = hubspot_df.rename(columns={'Created Date': 'Start Date'})

    if 'Expiration Date' in hubspot_df.columns:
        hubspot_df['Expiration Date'] = pd.to_datetime(hubspot_df['Expiration Date'], errors='coerce').dt.strftime('%Y-%m-%d').fillna('')

    # Calculate all core boolean flags and add Email Domain
    hubspot_df = _calculate_all_membership_flags(hubspot_df)

    # Use the new helper function to generate the tags list
    hubspot_df['Membership Status Tags List'] = hubspot_df.apply(_generate_membership_status_tags_list, axis=1)

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
        'City',
        'State/Regi',   # State/Region
        'Membersh',     # Original Membership field
        'Sales Doc',
        'Original Purchase Method', # Explicitly remove now that its purpose is served in tagging
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
        'Contact Email', # Keep original Contact Email
        'Cust ID', 
        'Email Domain', # Single Email Domain column
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

            original_count = len(membership_df)

            # Perform deduplication on membership data
            deduplicated_df = deduplicate_memberships(membership_df)
            deduplicated_count = len(deduplicated_df)

            # Process orders and add value tiers
            processed_df = process_orders_and_add_value_tiers(deduplicated_df, orders_df)

            # Categorize and sort by expiration status
            final_df_all = categorize_and_sort_memberships(processed_df)

            # --- Phase 1: Split into corporate and personal emails, apply recasing for personal ---
            personal_output_filename = None # Initialize to None

            if 'Contact Email' in final_df_all.columns and 'Customer Name' in final_df_all.columns:
                final_df_all['Email Domain'] = final_df_all['Contact Email'].apply(_extract_domain_from_email)

                corporate_mask = ~final_df_all['Email Domain'].isin(COMMON_FREE_EMAIL_DOMAINS)

                corporate_output_df = final_df_all[corporate_mask].copy()
                personal_output_df = final_df_all[~corporate_mask].copy()

                # Apply recasing to Customer Name for personal emails
                if not personal_output_df.empty:
                    personal_output_df['Customer Name'] = personal_output_df['Customer Name'].apply(recase_company_name)

                    # Define columns for personal email output (simplified for HubSpot import)
                    personal_export_cols = [
                        'Customer Name', 'Contact Email', 'Cust ID', 
                        'Total Order Value', 'Estimated Savings'
                    ]
                    personal_output_df = personal_output_df[[col for col in personal_export_cols if col in personal_output_df.columns]]

                    # Save personal emails to a separate CSV
                    temp_dir = tempfile.gettempdir()
                    personal_output_filename = f"common_email_prospects_{secure_filename(membership_file.filename).rsplit('.', 1)[0]}.csv"
                    personal_output_path = os.path.join(temp_dir, personal_output_filename)
                    personal_output_df.to_csv(personal_output_path, index=False)

                # The main processing continues with corporate_output_df
                final_df = corporate_output_df.drop(columns=['Email Domain'], errors='ignore')
            else:
                # If no email column or name column, proceed with original final_df_all
                final_df = final_df_all.drop(columns=['Email Domain'], errors='ignore') # Ensure Email Domain is dropped if it was added

            # Now apply export format to the 'corporate' part (or the whole df if no split occurred)
            if export_format == 'hubspot':
                final_df = format_for_hubspot_export(final_df)
                final_df = final_df.dropna(axis=1, how='all')
                file_extension = 'csv'
                output_filename = f"hubspot_{secure_filename(membership_file.filename).rsplit('.', 1)[0]}.csv"
            else: # Standard Excel output
                final_df = add_individual_boolean_tags_for_excel(final_df)
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
                                   deduplicated_count=deduplicated_count, # This count should reflect the total deduplicated before split
                                   reduction=original_count - deduplicated_count,
                                   filename=output_filename,
                                   personal_filename=personal_output_filename, # Pass the new filename
                                   export_format=export_format,
                                   preview_data=final_df.head(10).to_html(classes='table table-striped', index=False))

        except Exception as e:
            flash(f'Error processing files: {str(e)}')
            return redirect(url_for('upload_file'))
    else:
        flash('Invalid file type. Please upload Excel files (.xlsx or .xls)')
        return redirect(url_for('upload_file'))


@app.route('/process_phase_two', methods=['POST'])
def process_phase_two():
    # Check if orders file is present
    if 'orders_file' not in request.files:
        flash('Orders file is required for Phase 2 processing')
        return redirect(request.url)

    orders_file = request.files['orders_file']

    if orders_file.filename == '':
        flash('Orders file must be selected')
        return redirect(request.url)

    # Get export format for Phase 2
    export_format_phase_two = request.form.get('export_format_phase_two', 'standard_phase_two')

    if orders_file and allowed_file(orders_file.filename):
        try:
            # Read orders Excel file
            orders_df = pd.read_excel(orders_file)

            # Process phase two to identify non-WC prospects
            final_prospect_data, high_value_prospects = process_phase_two_orders(orders_df)

            # Save both files to temporary location
            temp_dir = tempfile.gettempdir()

            # Determine file extension and filename based on export format
            if export_format_phase_two == 'hubspot_phase_two':
                file_extension = 'csv'
                main_output_filename = f"hubspot_non_wc_prospects_{secure_filename(orders_file.filename).rsplit('.', 1)[0]}.csv"
                high_value_output_filename = f"hubspot_high_value_prospects_{secure_filename(orders_file.filename).rsplit('.', 1)[0]}.csv"

                # For HubSpot, we keep the data as-is with the Prospect Status Tags column
                # Both files already have the tags column from process_phase_two_orders
            else: # standard_phase_two
                file_extension = 'xlsx'
                main_output_filename = f"non_wc_prospects_{secure_filename(orders_file.filename).rsplit('.', 1)[0]}.xlsx"
                high_value_output_filename = f"high_value_prospects_{secure_filename(orders_file.filename).rsplit('.', 1)[0]}.xlsx"
                
                # For Excel, add individual boolean columns
                final_prospect_data = add_individual_boolean_tags_for_prospects_excel(final_prospect_data)
                high_value_prospects = add_individual_boolean_tags_for_prospects_excel(high_value_prospects)

            main_output_path = os.path.join(temp_dir, main_output_filename)
            high_value_output_path = os.path.join(temp_dir, high_value_output_filename)

            # Save both files
            if file_extension == 'csv':
                final_prospect_data.to_csv(main_output_path, index=False)
                high_value_prospects.to_csv(high_value_output_path, index=False)
            else:
                with pd.ExcelWriter(main_output_path, engine='openpyxl') as writer:
                    final_prospect_data.to_excel(writer, index=False, sheet_name='Non-WC Prospects')

                with pd.ExcelWriter(high_value_output_path, engine='openpyxl') as writer:
                    high_value_prospects.to_excel(writer, index=False, sheet_name='High-Value Prospects')

            return render_template('results.html', 
                                   original_count=len(orders_df),
                                   deduplicated_count=len(final_prospect_data),
                                   reduction=len(orders_df) - len(final_prospect_data),
                                   filename=main_output_filename,
                                   high_value_filename=high_value_output_filename,
                                   export_format=export_format_phase_two, # Pass the new export format
                                   preview_data=final_prospect_data.head(10).to_html(classes='table table-striped', index=False))

        except Exception as e:
            flash(f'Error processing Phase 2 file: {str(e)}')
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
