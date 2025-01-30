import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance, ratio
import pickle
import warnings

warnings.filterwarnings('ignore')


# # General Validation

# In[3]:


state_list = [
    
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID',
    'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
    'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
    'WI', 'WY'
]

terr_list = ['DC', 'GU', 'MH', 'MP', 'PR', 'VI']


# In[4]:


dict_dl_formats_new_mod={
    
    'AL': r'(^\d{7}$)|(^\d{8}$)',
    'AK': r'(^\d{7}$)|(^\d{6}$)|(^\d{5}$)|(^\d{4}$)|(^\d{3}$)|(^\d{2}$)|(^\d{1}$)',
    'AZ': r'(^\d{9}$)|(^\D{1}\d{8}$)', #same
    'AR': r'(^9\d{8}$)', #same
    'CA': r'(^\D{1}\d{7}$)',
    'CO': r'(^\d{2}-\d{3}-\d{4}$)',#same, but possible to modify
    'CT': r'(^[0][1-9]\d{7}$)|(^[1][0-9]\d{7}$)|(^[2][0-4]\d{7}$)',
    'DE': r'(^\d{7}$)|(^\d{6}$)|(^\d{5}$)|(^\d{4}$)|(^\d{3}$)|(^\d{2}$)|(^\d{1}$)',
    'FL': r'(^\D{1}\d{12}$)|(^\D{1}\d{3}-\d{3}-\d{3}-\d{3}$)|(^\D{1}-\d{3}-\d{3}-\d{3}-\d{3}$)',#same, but possible to modify
    'GA': r'(^\d{9}$)|(^\d{8}$)|(^\d{7}$)|(^\d{6}$)|(^\d{5}$)|(^\d{4}$)|(^\d{3}$)|(^\d{2}$)|(^\d{1}$)',
    'HI': r'(^[H]\d{8}$)',
    'ID': r'(^\D{2}\d{6}\D{1}$)',
    'IL': r'(^\D{1}\d{3}-\d{4}-\d{4}$)|(^\d{4}-\d{4}-\d{3}\D{1}-$)|(^\D{1}\d{11}$)|(^\d{11}\D{1}$)',#same, but possible to modify
    'IN': r'(^\d{4}-\d{2}-\d{4}$)',#same, but possible to modify
    'IA': r'(^\d{3}\D{2}\d{4}$)',
    'KS': r'(^[K]\d{8}$)',#r'(^\D{1}\d{2}-\d{2}-\d{4}$)|(^\D{1}\d{2} \d{2} \d{4}$)',---possible to modify
    'KY': r'(^\D{1}\d{2}-\d{3}-\d{3}$)|(^\D{1}\d{2} \d{3} \d{3}$)','LA': r'(^[0][0-1]\d{7}$)',
    'ME': r'(^\d{7}$)', #same
    'MD': r'(^\D{1}\d{12}$)|(^\D{1}-\d{3}-\d{3}-\d{3}-\d{3}$)|(^\D{1} \d{3} \d{3} \d{3} \d{3}$)',
    'MA': r'(^[S]\d{8}$)|(^[SA]\d{7}$)',#r'(^\D{1}\d{9}$)',
    'MI': r'(^\D{1}\d{12}$)|(^\D{1} \d{3} \d{3} \d{3} \d{3}$)|(^\D{1}\d{3} \d{3} \d{3} \d{3}$)|(^\D{1}\d{3}-\d{3}-\d{3}-\d{3}$)',
    'MN': r'(^\D{1}\d{12}$)|(^\D{1} \d{3} \d{3} \d{3} \d{3}$)|(^\D{1}\d{3} \d{3} \d{3} \d{3}$)|(^\D{1}\d{3}-\d{3}-\d{3}-\d{3}$)',
    'MS': r'(^\d{9}$)',#r'(^\d{3}-\d{2}-\d{4}$)',
    'MO': r'(^\D{1}\d{9}$)|(^\D{1}\d{8}$)|(^\D{1}\d{7}$)|(^\D{1}\d{6}$)|(^\d{3}\D{1}\d{6}$)',
    'MT': r'(^[0][1-9]\d{3}[1][9]\d{2}[4][1][0-3][0-9]$)|(^[1][0-2]\d{3}[1][9]\d{2}[4][1][0-3][0-9]$)|(^[0][1-9]\d{3}[2][0]\d{2}[4][1][0-3][0-9]$)|(^[1][0-2]\d{3}[2][0]\d{2}[4][1][0-3][0-9]$)', #r'(^\d{9}$)',  #additional conditions
    'NE': r'(^[A]\d{8}$)|(^[B]\d{8}$)|(^[C]\d{8}$)|(^[E]\d{8}$)|(^[G]\d{8}$)|(^[H]\d{8}$)|(^[V]\d{8}$)',
    'NV': r'(^\d{10}$)|(^\d{12}$)',
    'NH': r'(^[NHL]\d{8}$)|(^[0][1-9]\D{3}\d{2}[0-3][0-9]\d{1}$)|(^[1][0-2]\D{3}\d{2}[0-3][0-9]\d{1}$)', #r'(^\d{10}$)',  #additional condition
    'NJ': r'(^\D{1}\d{4}-\d{5}-[0][1-9]\d{3}$)|(^\D{1}\d{4} \d{5} [0][1-9]\d{3})|(^\D{1}\d{4}-\d{5}-[1][0-2]\d{3}$)|(^\D{1}\d{4} \d{5} [1][0-2]\d{3})|(^\D{1}\d{4}-\d{5}-[5][1-9]\d{3}$)|(^\D{1}\d{4} \d{5} [5][1-9]\d{3}$)|(^\D{1}\d{4}-\d{5}-[6][0-2]\d{3}$)|(^\D{1}\d{4} \d{5} [6][0-2]\d{3}$)', #r'(^\D{1}\d{14}$)|(^\D{1}\d{4}-\d{5}-\d{5}$)|(^\D{1}\d{4} \d{5} \d{5}$)',  #additional conditions
    'NM': r'(^\d{9}$)',
    'NY': r'(^\d{9}$)|(^\d{3} \d{3} \d{3}$)',#same, but possible to modify  #r'(^\d{9}$)|(^\d{3} \d{3} \d{3}$)',
    'NC': r'(^\d{12}$)|(^\d{11}$)|(^\d{10}$)|(^\d{9}$)|(^\d{8}$)|(^\d{7}$)|(^\d{6}$)|(^\d{5}$)|(^\d{4}$)|(^\d{3}$)|(^\d{2}$)|(^\d{1}$)',
    'ND': r'(^\D{3}-\d{2}-\d{4}$)|(^\D{3}\d{2}\d{4}$)|(^\D{3} \d{2} \d{4}$)',#same (modified)
    'OH': r'(^\d{8}$)|(^\D{2}\d{6}$)',
    'OK': r'(^\D{1}\d{9}$)',
    'OR': r'(^\D{1}\d{6}$)|(^\d{7}$)|(^\d{6}$)|(^\d{5}$)|(^\d{4}$)|(^\d{3}$)|(^\d{2}$)|(^\d{1}$)',
    'PA': r'(^\d{2} \d{3} \d{3}$)|(^\d{8}$)', #same, but possible to modify
    'RI': r'(^\d{7}$)|(^\d{8}$)(^[V]\d{6}$)',#additional conditions
    'SC': r'(^\d{9}$)',
    'SD': r'(^\d{8}$)',
    'TN': r'(^\d{9}$)',#r'(^\d{7}$)|(^\d{8}$)|(^\d{9}$)',
    'TX': r'(^\d{8}$)|(^\d{7}$)',
    'UT': r'(^\d{9}$)|(^\d{8}$)|(^\d{7}$)|(^\d{6}$)|(^\d{5}$)|(^\d{4}$)',
    'VT': r'(^\d{8}$)|(^\d{7}[A]$)',
    'VA': r'(^\D{1}\d{8}$)|(^\D{1}\d{2}-\d{2}-\d{4}$)|(^\D{1}\d{2} \d{2} \d{4}$)',#same (modified)
    'WA': r'(^\D{3}[*][*]\D{2}\d{3}\D{1}\d{1}$)|(^[WDL]\D{2}\d{3}\D{1}\d{1}$)',# ?????  #r'(^\D{3}[*][*]\D{2}\d{3}\D{1}\d{1}$)|(^\D{3}\D{2}\d{3}\D{1}\d{1}$)',
    'DC': r'(^\d{7}$)',
    'WV': r'(^\d{7}$)|(^\D{1}\d{6}$)',
    'WI': r'(^\D{1}\d{3}-\d{4}-\d{4}-\d{2}$)|(^\D{1}\d{3} \d{4} \d{4} \d{2}$)',    #same, but possible to modify    r'(^\D{1}\d{13}$)|(^\D{1}\d{3}-\d{4}-\d{4}-\d{2}$)|(^\D{1}\d{3} \d{4} \d{4} \d{2}$)',
    'WY': r'(^\d{6}-\d{3}$)|(^\d{2}-\d{7}$)'#same, but possible to modify #r'(^\d{6}-\d{3}$)|(^\d{9}$)'
}


# In[5]:


list_valid_phone_prefix = [
    
    201, 202, 203, 205, 206, 207, 208, 209, 210, 212, 213, 214, 215, 216, 217,
    218, 219, 220, 223, 224, 225, 228, 229, 231, 234, 239, 240, 248, 251, 252,
    253, 254, 256, 260, 262, 263, 267, 269, 270, 272, 276, 279, 281, 401, 402,
    404, 405, 406, 407, 408, 409, 410, 412, 413, 414, 415, 417, 419, 423, 424,
    425, 430, 432, 434, 435, 440, 442, 443, 445, 447, 448, 458, 463, 464, 468,
    469, 470, 472, 474, 475, 478, 479, 480, 484, 500, 501, 502, 503, 504, 505,
    507, 508, 509, 510, 512, 513, 515, 516, 517, 518, 520, 521, 522, 523, 524,
    525, 526, 527, 528, 529, 530, 531, 533, 534, 539, 540, 541, 544, 551, 557,
    559, 561, 562, 563, 564, 566, 567, 570, 571, 572, 573, 574, 575, 577, 580,
    582, 584, 585, 586, 588, 601, 602, 603, 605, 606, 607, 608, 609, 610, 612,
    614, 615, 616, 617, 618, 619, 620, 623, 626, 628, 629, 630, 631, 636, 640,
    641, 646, 650, 651, 656, 657, 658, 659, 660, 661, 662, 667, 669, 672, 678,
    680, 681, 682, 683, 689, 700, 701, 702, 703, 704, 706, 707, 708, 710, 712,
    713, 714, 715, 716, 717, 718, 719, 720, 724, 725, 726, 727, 731, 732, 734,
    737, 740, 742, 743, 747, 753, 754, 757, 760, 762, 763, 765, 769, 770, 771,
    772, 773, 774, 775, 779, 781, 785, 786, 800, 801, 802, 803, 804, 805, 806,
    808, 810, 812, 813, 814, 815, 816, 817, 818, 820, 826, 828, 830, 831, 832,
    833, 835, 838, 839, 840, 843, 844, 845, 847, 848, 850, 854, 855, 856, 857,
    858, 859, 860, 862, 863, 864, 865, 866, 870, 872, 877, 878, 888, 900, 901,
    903, 904, 906, 907, 908, 909, 910, 912, 913, 914, 915, 916, 917, 918, 919,
    920, 925, 928, 929, 930, 931, 934, 936, 937, 938, 940, 941, 943, 945, 947,
    948, 949, 951, 952, 954, 956, 959, 970, 971, 972, 973, 978, 979, 980, 983,
    984, 985, 986, 989
]

list_valid_phone_prefix = [str(i) for i in list_valid_phone_prefix]


# In[6]:


def ssn_validation(df, ssn_field_name='SSN'):
    """
    Validates the Social Security Numbers (SSNs) in a DataFrame column by checking their format.

    This function processes the SSN column by:
    - Removing hyphens from SSNs.
    - Converting the SSNs to numeric values.
    - Checking if the length of each SSN is exactly 9 digits.

    Args:
        df (pd.DataFrame): The DataFrame containing the SSNs to validate.
        ssn_field_name (str): The name of the column containing SSNs. Defaults to 'SSN'.

    Returns:
        list: A list of boolean values indicating whether each SSN is valid (True) or invalid (False).

    Example:
        >>> import pandas as pd
        >>> data = {'SSN': ['123-45-6789', '987-65-432', '111-22-3333']}
        >>> df = pd.DataFrame(data)
        >>> valid_ssns = ssn_validation(df)
        >>> print(valid_ssns)
        [True, False, True]
    """
    df_temp = df.copy(deep=True)

    # Remove hyphens, convert to numeric, and fill invalid entries with 0
    df_temp[ssn_field_name] = pd.to_numeric(
        df_temp[ssn_field_name].astype(str).str.replace(
            "-", "")).fillna(0).astype(int)

    # Calculate the length of the processed SSNs
    df_temp['ssn_length'] = (pd.to_numeric(
        df_temp[ssn_field_name]).fillna(0).astype(int).astype(str).str.len())

    # Determine if each SSN is valid based on length
    df_temp['ssn_valid'] = (df_temp['ssn_length'] == 9)

    return df_temp['ssn_valid'].tolist()


# In[7]:


def routing_number_validation(df, routing_field_name='RoutingNumber'):
    """
    Validates routing numbers in a DataFrame column based on their length and checksum.

    This function checks:
    - If routing numbers have a valid length (8 or 9 digits).
    - If the checksum of the routing number is valid based on standard routing number validation rules.

    Args:
        df (pd.DataFrame): The DataFrame containing the routing numbers to validate.
        routing_field_name (str): The name of the column containing routing numbers. Defaults to 'RoutingNumber'.

    Returns:
        list: A list of boolean values indicating whether each routing number is valid (True) or invalid (False).

    Example:
        >>> import pandas as pd
        >>> data = {'RoutingNumber': ['12345678', '876543219', '123456789']}
        >>> df = pd.DataFrame(data)
        >>> valid_routing = routing_number_validation(df)
        >>> print(valid_routing)
        [False, True, True]
    """
    df_temp = df.copy(deep=True)

    # Calculate the length of the routing numbers
    df_temp['routing_number_length'] = pd.to_numeric(
        df_temp[routing_field_name],
        errors='coerce').fillna(0).astype(int).astype(str).str.len()

    # Initialize validity column
    df_temp['routing_number_valid'] = False

    # Check validity for 8-digit routing numbers
    df_temp.loc[(df_temp['routing_number_length'] == 8) & ((10 - (
        3 *
        (pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
        ) == 8][routing_field_name].astype(str).str[2],
                       errors='coerce') +
         pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
         ) == 8][routing_field_name].astype(str).str[5],
                       errors='coerce')) + 7 *
        (pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
        ) == 8][routing_field_name].astype(str).str[0],
                       errors='coerce') +
         pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
         ) == 8][routing_field_name].astype(str).str[3],
                       errors='coerce') +
         pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
         ) == 8][routing_field_name].astype(str).str[6],
                       errors='coerce')) + 1 *
        (pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
        ) == 8][routing_field_name].astype(str).str[1],
                       errors='coerce') +
         pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
         ) == 8][routing_field_name].astype(str).str[4],
                       errors='coerce'))
    ) % 10) % 10 == pd.to_numeric(df_temp[df_temp[routing_field_name].astype(
        str).str.len() == 8][routing_field_name].astype(str).str[7],
                                  errors='coerce')),
                'routing_number_valid'] = True

    # Check validity for 9-digit routing numbers
    df_temp.loc[(df_temp['routing_number_length'] == 9) & ((10 - (
        3 *
        (pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
        ) == 9][routing_field_name].astype(str).str[0],
                       errors='coerce') +
         pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
         ) == 9][routing_field_name].astype(str).str[3],
                       errors='coerce') +
         pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
         ) == 9][routing_field_name].astype(str).str[6],
                       errors='coerce')) + 7 *
        (pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
        ) == 9][routing_field_name].astype(str).str[1],
                       errors='coerce') +
         pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
         ) == 9][routing_field_name].astype(str).str[4],
                       errors='coerce') +
         pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
         ) == 9][routing_field_name].astype(str).str[7],
                       errors='coerce')) + 1 *
        (pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
        ) == 9][routing_field_name].astype(str).str[2],
                       errors='coerce') +
         pd.to_numeric(df_temp[df_temp[routing_field_name].astype(str).str.len(
         ) == 9][routing_field_name].astype(str).str[5],
                       errors='coerce'))
    ) % 10) % 10 == pd.to_numeric(df_temp[df_temp[routing_field_name].astype(
        str).str.len() == 9][routing_field_name].astype(str).str[8],
                                  errors='coerce')),
                'routing_number_valid'] = True

    return df_temp['routing_number_valid'].tolist()


# In[8]:


def account_bank_number_validation(df, aba_field_name='AccountNumber'):
    """
    Validates bank account numbers in a DataFrame column by checking their lengths.

    This function determines if an account number is valid based on the following criteria:
    - The account number length must be between 5 and 17 digits (inclusive).

    Args:
        df (pd.DataFrame): The DataFrame containing the account numbers to validate.
        aba_field_name (str): The name of the column containing account numbers. Defaults to 'AccountNumber'.

    Returns:
        list: A list of boolean values indicating whether each account number is valid (True) or invalid (False).

    Example:
        >>> import pandas as pd
        >>> data = {'AccountNumber': ['12345', '12345678901234567', '1234']}
        >>> df = pd.DataFrame(data)
        >>> valid_accounts = account_bank_number_validation(df)
        >>> print(valid_accounts)
        [True, True, False]
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_temp = df.copy(deep=True)

    # Calculate the length of the account numbers
    df_temp['account_number_length'] = pd.to_numeric(
        df_temp[aba_field_name], errors='coerce'
    ).fillna(0).astype(int).astype(str).str.len()

    # Determine validity based on length (5 to 17 digits)
    df_temp['account_number_valid'] = df_temp['account_number_length'].isin(list(range(5, 18)))

    # Return the validity as a list
    return df_temp['account_number_valid'].tolist()


# In[9]:


def phone_validation(df, phone_field_name='MobilePhone'):
    """
    Validates phone numbers in a DataFrame column based on specific criteria.

    This function performs the following checks:
    - Ensures the phone number has exactly 10 digits.
    - Validates the first three digits (area code) against a predefined list of valid prefixes.
    - Ensures the third digit of the phone number is not '0' or '1'.

    Args:
        df (pd.DataFrame): The DataFrame containing phone numbers to validate.
        phone_field_name (str): The name of the column containing phone numbers. Defaults to 'MobilePhone'.

    Returns:
        list: A list of boolean values indicating whether each phone number is valid (True) or invalid (False).

    Example:
        >>> import pandas as pd
        >>> list_valid_phone_prefix = ['123', '456', '789']
        >>> data = {'MobilePhone': ['1234567890', '9876543210', '4561230000']}
        >>> df = pd.DataFrame(data)
        >>> valid_phones = phone_validation(df)
        >>> print(valid_phones)
        [True, False, False]
    """
    # Create a deep copy of the DataFrame to avoid modifying the original
    df_temp = df.copy(deep=True)

    # Initialize a column for phone number validity
    df_temp['phone_valid'] = False

    # Ensure phone numbers are treated as strings
    df_temp['phone'] = df_temp[phone_field_name].astype(str)

    # Extract the first three digits (area code) for validation
    df_temp['0-2 digits'] = (
        pd.to_numeric(
            df_temp['phone'].str.replace('(', '')
                             .replace(')', '')
                             .replace(' ', ''),
            errors='coerce'
        )
        .fillna(0)
        .astype(float)
        .astype(str)
        .str[:3]
    )

    # Extract the third digit of the phone number
    df_temp['3rd digit'] = pd.to_numeric(
        df_temp['phone'].str.replace('(', '')
                         .replace(')', '')
                         .replace(' ', ''),
        errors='coerce'
    ).fillna(0).astype(float).astype(str).str[3]

    # Calculate the length of each phone number
    df_temp['phone_length'] = df_temp['phone'].str.len()

    # Validate phone numbers based on criteria
    df_temp['valid'] = (
        (df_temp['0-2 digits'].isin(list_valid_phone_prefix)) &
        (~df_temp['3rd digit'].isin(['0', '1'])) &
        (df_temp['phone_length'] == 10)
    )

    # Create a list of valid phone numbers
    list_valid_phone = df_temp[df_temp['valid']]['phone'].tolist()

    # Mark valid phone numbers in the DataFrame
    df_temp.loc[df_temp['phone'].isin(list_valid_phone), 'phone_valid'] = True
    df_temp.loc[df_temp['phone_valid'].isna(), 'phone_valid'] = False

    # Return the validity status as a list
    return df_temp['phone_valid'].tolist()


# In[10]:


def state_validation(df, state_field_name='State'):
    """
    Validates state values in a DataFrame column by checking them against valid U.S. states and territories.

    This function checks if the values in the specified column are part of a predefined list of valid U.S. states
    and territories.

    Args:
        df (pd.DataFrame): The DataFrame containing the state values to validate.
        state_field_name (str): The name of the column containing state values. Defaults to 'State'.

    Returns:
        list: A list of boolean values indicating whether each state value is valid (True) or invalid (False).

    Example:
        >>> import pandas as pd
        >>> state_list = ['CA', 'NY', 'TX']  # Example valid states
        >>> terr_list = ['PR', 'GU']  # Example valid territories
        >>> data = {'State': ['CA', 'PR', 'ZZ']}
        >>> df = pd.DataFrame(data)
        >>> valid_states = state_validation(df)
        >>> print(valid_states)
        [True, True, False]
    """
    # Create a deep copy of the DataFrame to avoid modifying the original
    df_temp = df.copy(deep=True)

    # Initialize a column to indicate state validity
    df_temp['state_valid'] = False

    # Mark states as valid if they are in the predefined state or territory lists
    df_temp.loc[df_temp[state_field_name].isin(state_list + terr_list), 'state_valid'] = True

    # Return the validity status as a list
    return df_temp['state_valid'].tolist()


# In[11]:


def driver_license_validation(
        df,
        dl_field_name='DrivingLicenseNumber',
        dl_state_field_name='DrivingLicenseIssuingState'):
    """
    Validates driver license numbers and issuing states in a DataFrame.

    This function performs two types of validations:
    1. State Validation:
       - Checks if the issuing state of the driver license is valid, based on a predefined list of U.S. states and territories.
    2. Driver License Format Validation:
       - Validates the driver license number format based on state-specific rules provided in `dict_dl_formats_new_mod`.

    Args:
        df (pd.DataFrame): The DataFrame containing the driver license details to validate.
        dl_field_name (str): The name of the column containing driver license numbers. Defaults to 'DrivingLicenseNumber'.
        dl_state_field_name (str): The name of the column containing driver license issuing states. Defaults to 'DrivingLicenseIssuingState'.

    Returns:
        tuple: A tuple of two lists:
            - A list of boolean values indicating whether each driver license is valid (True) or invalid (False).
            - A list of boolean values indicating whether the issuing state is valid (True) or invalid (False).

    Example:
        >>> import pandas as pd
        >>> state_list = ['CA', 'NY', 'TX']
        >>> terr_list = ['PR', 'GU']
        >>> dict_dl_formats_new_mod = {
        ...     'CA': r'^[A-Z]{1}[0-9]{7}$',
        ...     'NY': r'^[A-Z]{3}[0-9]{9}$',
        ...     'TX': r'^[0-9]{8}$'
        ... }
        >>> data = {
        ...     'DrivingLicenseNumber': ['A1234567', 'ABC123456789', '12345678'],
        ...     'DrivingLicenseIssuingState': ['CA', 'NY', 'TX']
        ... }
        >>> df = pd.DataFrame(data)
        >>> dl_valid, dl_state_valid = driver_license_validation(df)
        >>> print(dl_valid)
        [True, True, True]
        >>> print(dl_state_valid)
        [True, True, True]
    """
    # Create a deep copy of the DataFrame to avoid modifying the original
    df_temp = df.copy(deep=True)

    # Initialize validity columns
    df_temp['dl_state_valid'] = False
    df_temp['dl_valid'] = False

    # Validate issuing states
    df_temp.loc[df_temp[dl_state_field_name].isin(state_list + terr_list), 'dl_state_valid'] = True

    # Validate driver license numbers based on state-specific formats
    for state, format_regex in dict_dl_formats_new_mod.items():
        df_temp.loc[
            (df_temp[dl_field_name].astype(str).str.contains(format_regex, regex=True)) &
            (df_temp[dl_state_field_name] == state),
            'dl_valid'
        ] = True

    # Return driver license validity and state validity as separate lists
    return df_temp['dl_valid'].tolist(), df_temp['dl_state_valid'].tolist()


# In[12]:


def address_validation(df, address_benchmark_field_name='Subject1_ReturnCode'):
    """
    Validates addresses in a DataFrame based on a benchmark field's return codes.

    The function performs the following validation:
    - Sets address validity to 0.5 for return codes starting with specific patterns ('22', '32').
    - Sets address validity to 1 for return codes starting with '31'.
    - Defaults address validity to 0 for all other cases.

    Args:
        df (pd.DataFrame): The DataFrame containing the address return codes to validate.
        address_benchmark_field_name (str): The name of the column containing return codes. Defaults to 'Subject1_ReturnCode'.

    Returns:
        list: A list of numerical values indicating the address validity:
              - 0 for invalid addresses.
              - 0.5 for partially valid addresses.
              - 1 for fully valid addresses.

    Example:
        >>> import pandas as pd
        >>> data = {'Subject1_ReturnCode': ['A31', 'B22', 'C32', 'D44']}
        >>> df = pd.DataFrame(data)
        >>> address_validity = address_validation(df)
        >>> print(address_validity)
        [1, 0.5, 0.5, 0]
    """
    # Create a deep copy of the DataFrame to avoid modifying the original
    df_temp = df.copy(deep=True)

    # Initialize the address validity column to 0 (default invalid state)
    df_temp['address_valid'] = 0

    # Set validity to 0.5 for codes ending in '22' or '32'
    df_temp.loc[df_temp[address_benchmark_field_name].str[1:].isin(['22', '32']), 'address_valid'] = 0.5

    # Set validity to 1 for codes ending in '31'
    df_temp.loc[df_temp[address_benchmark_field_name].str[1:] == '31', 'address_valid'] = 1

    # Return the address validity as a list
    return df_temp['address_valid'].tolist()


# In[13]:


def dob_validation(df, dob_field_name='DateOfBirth'):
    """
    Validates date of birth (DOB) values in a DataFrame.

    This function checks if the date of birth field is valid:
    - If the field is null (NaN), it is marked as invalid.
    - If the field is not null, it is marked as valid.

    Args:
        df (pd.DataFrame): The DataFrame containing date of birth values to validate.
        dob_field_name (str): The name of the column containing date of birth values. Defaults to 'DateOfBirth'.

    Returns:
        list: A list of boolean values indicating whether each date of birth is valid (True) or invalid (False).

    Example:
        >>> import pandas as pd
        >>> data = {'DateOfBirth': ['1990-01-01', None, '2000-12-31']}
        >>> df = pd.DataFrame(data)
        >>> dob_validity = dob_validation(df)
        >>> print(dob_validity)
        [True, False, True]
    """
    # Create a deep copy of the DataFrame to avoid modifying the original
    df_temp = df.copy(deep=True)

    # Initialize the dob_valid column as True
    df_temp['dob_valid'] = True

    # Mark rows as invalid (False) where the date of birth is NaN
    df_temp.loc[df_temp[dob_field_name].isna(), 'dob_valid'] = False

    # Return the validity as a list
    return df_temp['dob_valid'].tolist()


# In[14]:


def pii_validation_score(df):
    """
    Calculates a PII (Personally Identifiable Information) validation score for each record in a DataFrame.

    This function computes a score based on the validity of various PII attributes:
    - Driver License Validity (`dl_valid`): 30 points
    - Social Security Number Validity (`ssn_valid`): 30 points
    - Phone Number Validity (`phone_valid`): 15 points
    - State Validity (`state_valid`): 5 points
    - Driver License Issuing State Validity (`dl_state_valid`): 5 points
    - Date of Birth Validity (`dob_valid`): 5 points
    - Address Validity (`address_valid`): 10 points

    The total score is calculated as a weighted sum of these attributes, rounded to 2 decimal places.

    Args:
        df (pd.DataFrame): The DataFrame containing the PII validation columns.

    Returns:
        list: A list of scores (floats) representing the PII validation score for each record.

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'dl_valid': [1, 0],
        ...     'ssn_valid': [1, 1],
        ...     'phone_valid': [0, 1],
        ...     'state_valid': [1, 1],
        ...     'dl_state_valid': [1, 0],
        ...     'dob_valid': [1, 1],
        ...     'address_valid': [0.5, 1],
        ... }
        >>> df = pd.DataFrame(data)
        >>> scores = pii_validation_score(df)
        >>> print(scores)
        [85.5, 91.0]
    """
    # Create a deep copy of the DataFrame to avoid modifying the original
    df_temp = df.copy(deep=True)

    # Calculate the PII validation score based on weights
    df_temp['pii_validation_score'] = (
        30 * df_temp['dl_valid'] +
        30 * df_temp['ssn_valid'] +
        15 * df_temp['phone_valid'] +
        5 * df_temp['state_valid'] +
        5 * df_temp['dl_state_valid'] +
        5 * df_temp['dob_valid'] +
        10 * df_temp['address_valid']
    )

    # Return the scores rounded to 2 decimal places as a list
    return df_temp['pii_validation_score'].round(2).tolist()


# In[15]:


def bank_info_validation_score(df):
    """
    Calculates a bank information validation score for each record in a DataFrame.

    This function computes a score based on the validity of bank-related information:
    - Routing Number Validity (`routing_number_valid`): 80 points
    - Account Number Validity (`account_number_valid`): 20 points

    The total score is calculated as a weighted sum of these attributes, rounded to 2 decimal places.

    Args:
        df (pd.DataFrame): The DataFrame containing the bank validation columns.

    Returns:
        list: A list of scores (floats) representing the bank information validation score for each record.

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'routing_number_valid': [1, 0],
        ...     'account_number_valid': [1, 1],
        ... }
        >>> df = pd.DataFrame(data)
        >>> scores = bank_info_validation_score(df)
        >>> print(scores)
        [100.0, 20.0]
    """
    # Create a deep copy of the DataFrame to avoid modifying the original
    df_temp = df.copy(deep=True)

    # Calculate the bank information validation score based on weights
    df_temp['bank_info_validation_score'] = (
        80 * df_temp['routing_number_valid'] +
        20 * df_temp['account_number_valid']
    )

    # Return the scores rounded to 2 decimal places as a list
    return df_temp['bank_info_validation_score'].round(2).tolist()


# # Similarity Measures

# In[16]:


def jaccard_similarity(a, b):
    """
    Calculates the Jaccard similarity between two sets.

    Args:
        a (iterable): First iterable to compare.
        b (iterable): Second iterable to compare.

    Returns:
        float: Jaccard similarity score between the two sets.
    """
    a = set(a)
    b = set(b)
    return float(len(a.intersection(b))) / len(a.union(b))


def jaccard_similarity_measure(df, base_field_name_1, base_field_name_2,
                               append_field_name_1, append_field_name_2):
    """
    Computes Jaccard similarity measures for substrings in specified columns of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        base_field_name_1 (str): Name of the first base field to compare.
        base_field_name_2 (str): Name of the second base field to compare (can be an empty string).
        append_field_name_1 (str): Name of the first append field to compare.
        append_field_name_2 (str): Name of the second append field to compare.

    Returns:
        list: A list of Jaccard similarity scores for each row in the DataFrame.

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'BaseField1': ['abc', 'xyz'],
        ...     'BaseField2': ['def', 'uvw'],
        ...     'AppendField1': ['ghi', 'rst'],
        ...     'AppendField2': ['jkl', 'opq']
        ... }
        >>> df = pd.DataFrame(data)
        >>> scores = jaccard_similarity_measure(df, 'BaseField1', 'BaseField2', 'AppendField1', 'AppendField2')
        >>> print(scores)
        [0.0, 0.0]
    """
    df_temp = df.copy(deep=True)
    list_jaccard_similarity_scores = []

    def generate_substrings(s):
        """Generates all substrings of a string."""
        return [
            s[i:j] for i in range(len(s)) for j in range(i + 1,
                                                         len(s) + 1)
        ]

    if base_field_name_2 == '':
        # Loop through rows when only one base field is used
        for i, j, k in zip(
                df_temp[base_field_name_1],
                df_temp[append_field_name_1].fillna("0").str.split().str[0],
                df_temp[append_field_name_2].fillna("0").str.split().str[0],
        ):
            i, j, k = str(i), str(j), str(k)
            substr_i, substr_j, substr_k = generate_substrings(
                i), generate_substrings(j), generate_substrings(k)
            score = max(jaccard_similarity(substr_i, substr_j),
                        jaccard_similarity(substr_i, substr_k))
            list_jaccard_similarity_scores.append(score)
    else:
        # Loop through rows when two base fields are used
        for i, j, k, l in zip(
                df_temp[base_field_name_1].fillna("0").str.split().str[0],
                df_temp[base_field_name_2].fillna("0").str.split().str[0],
                df_temp[append_field_name_1].fillna("1").str.split().str[0],
                df_temp[append_field_name_2].fillna("1").str.split().str[0],
        ):
            i, j, k, l = str(i), str(j), str(k), str(l)
            substr_i, substr_j, substr_k, substr_l = (
                generate_substrings(i),
                generate_substrings(j),
                generate_substrings(k),
                generate_substrings(l),
            )
            score = max(
                jaccard_similarity(substr_i, substr_k),
                jaccard_similarity(substr_i, substr_l),
                jaccard_similarity(substr_j, substr_k),
                jaccard_similarity(substr_j, substr_l),
            )
            list_jaccard_similarity_scores.append(score)

    return list_jaccard_similarity_scores


# In[17]:


def levenshtein_similarity_measure(df, base_field_name_1, base_field_name_2, append_field_name_1, append_field_name_2):
    """
    Computes Levenshtein similarity measures for specified columns in a DataFrame.

    This function calculates the similarity between strings in two sets of columns using
    the Levenshtein similarity ratio. The similarity is calculated as:
    - The maximum similarity ratio between strings from the base fields and append fields.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        base_field_name_1 (str): Name of the first base field to compare.
        base_field_name_2 (str): Name of the second base field to compare (can be an empty string).
        append_field_name_1 (str): Name of the first append field to compare.
        append_field_name_2 (str): Name of the second append field to compare.

    Returns:
        list: A list of similarity scores (floats) for each row in the DataFrame.

    Example:
        >>> import pandas as pd
        >>> from Levenshtein import ratio
        >>> data = {
        ...     'BaseField1': ['apple', 'orange'],
        ...     'BaseField2': ['fruit', 'citrus'],
        ...     'AppendField1': ['applepie', 'grapefruit'],
        ...     'AppendField2': ['pineapple', 'oranges']
        ... }
        >>> df = pd.DataFrame(data)
        >>> scores = levenshtein_similarity_measure(df, 'BaseField1', 'BaseField2', 'AppendField1', 'AppendField2')
        >>> print(scores)
        [0.75, 0.88]
    """
    # Create a deep copy of the DataFrame to avoid modifying the original
    df_temp = df.copy(deep=True)

    # Initialize a list to store similarity measures
    list_levenshtein_similarity_measure = []

    if base_field_name_2 == '':
        # If only one base field is provided
        for i, j, k in zip(
            df_temp[base_field_name_1].astype(str),
            df_temp[append_field_name_1].astype(str).str.split().str[0],
            df_temp[append_field_name_2].astype(str).str.split().str[0]
        ):
            similarity = max(ratio(i, j), ratio(i, k))
            list_levenshtein_similarity_measure.append(similarity)
    else:
        # If two base fields are provided
        for i, j, k, l in zip(
            df_temp[base_field_name_1].astype(str),
            df_temp[base_field_name_2].astype(str),
            df_temp[append_field_name_1].astype(str),
            df_temp[append_field_name_2].astype(str)
        ):
            similarity = max(
                ratio(i, k), ratio(i, l),
                ratio(j, k), ratio(j, l)
            )
            list_levenshtein_similarity_measure.append(similarity)

    return list_levenshtein_similarity_measure


# In[18]:


def cosine_similarity_measure(df, base_field_name_1, base_field_name_2,
                              append_field_name_1, append_field_name_2):
    """
    Computes cosine similarity measures between substrings generated from specified fields in a DataFrame.

    This function calculates the similarity between strings in the base and append fields using cosine similarity
    with TF-IDF vectorization. The similarity is computed as:
    - The maximum cosine similarity score between substrings of the base fields and the append fields.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        base_field_name_1 (str): Name of the first base field to compare.
        base_field_name_2 (str): Name of the second base field to compare (can be an empty string).
        append_field_name_1 (str): Name of the first append field to compare.
        append_field_name_2 (str): Name of the second append field to compare.

    Returns:
        list: A list of cosine similarity scores for each row in the DataFrame.

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'BaseField1': ['abc', 'xyz'],
        ...     'BaseField2': ['def', 'uvw'],
        ...     'AppendField1': ['ghi', 'rst'],
        ...     'AppendField2': ['jkl', 'opq']
        ... }
        >>> df = pd.DataFrame(data)
        >>> scores = cosine_similarity_measure(df, 'BaseField1', 'BaseField2', 'AppendField1', 'AppendField2')
        >>> print(scores)
        [0.0, 0.0]
    """
    # Create a deep copy of the DataFrame to avoid modifying the original
    df_temp = df.copy(deep=True)
    list_cosine_similarity_measure = []

    def generate_substrings(s):
        """Generates all substrings of a string."""
        return ' '.join(
            [s[i:j] for i in range(len(s)) for j in range(i + 1,
                                                          len(s) + 1)])

    if base_field_name_2 == '':
        # Case with only one base field
        for i, j, k in zip(
                df_temp[base_field_name_1].fillna('0').astype(str),
                df_temp[append_field_name_1].fillna('0').str.split().str[0],
                df_temp[append_field_name_2].fillna('0').str.split().str[0]):
            res_i, res_j, res_k = generate_substrings(i), generate_substrings(
                j), generate_substrings(k)

            # Calculate cosine similarities
            data_ij = [res_i, res_j]
            data_ik = [res_i, res_k]
            vectorizer = TfidfVectorizer(tokenizer=lambda txt: txt.split())

            trsfm_ij = vectorizer.fit_transform(data_ij)
            trsfm_ik = vectorizer.fit_transform(data_ik)

            list_cosine_similarity_measure.append(
                max(
                    cosine_similarity(trsfm_ij[0:1], trsfm_ij)[0][1],
                    cosine_similarity(trsfm_ik[0:1], trsfm_ik)[0][1]))
    else:
        # Case with two base fields
        for i, j, k, l in zip(
                df_temp[base_field_name_1].fillna('0').astype(str),
                df_temp[base_field_name_2].fillna('0').astype(str),
                df_temp[append_field_name_1].fillna('1').astype(str),
                df_temp[append_field_name_2].fillna('1').astype(str)):
            res_i, res_j, res_k, res_l = (
                generate_substrings(i),
                generate_substrings(j),
                generate_substrings(k),
                generate_substrings(l),
            )

            # Calculate cosine similarities
            data_pairs = [[res_i, res_k], [res_i, res_l], [res_j, res_k],
                          [res_j, res_l]]
            max_similarity = 0
            for data in data_pairs:
                vectorizer = TfidfVectorizer(tokenizer=lambda txt: txt.split())
                trsfm = vectorizer.fit_transform(data)
                similarity = cosine_similarity(trsfm[0:1], trsfm)[0][1]
                max_similarity = max(max_similarity, similarity)

            list_cosine_similarity_measure.append(max_similarity)

    return list_cosine_similarity_measure


# In[19]:


def convex_similarity_measure(df, similarity_measure_jaccard_field, similarity_measure_levenshtein_field,
                              similarity_measure_cosine_field):
    """
    Computes a convex combination of similarity measures (Jaccard, Levenshtein, and Cosine) 
    and normalizes the result to a range between 0 and 1.

    The resulting similarity measure is then normalized to ensure values are between 0 and 1.

    Args:
        df (pd.DataFrame): The DataFrame containing the similarity measure fields.
        similarity_measure_jaccard_field (str): Column name for the Jaccard similarity measure.
        similarity_measure_levenshtein_field (str): Column name for the Levenshtein similarity measure.
        similarity_measure_cosine_field (str): Column name for the Cosine similarity measure.

    Returns:
        list: A list of normalized convex similarity scores for each record in the DataFrame.

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'JaccardSimilarity': [0.8, 0.6, 0.7],
        ...     'LevenshteinSimilarity': [0.75, 0.65, 0.7],
        ...     'CosineSimilarity': [0.9, 0.7, 0.8]
        ... }
        >>> df = pd.DataFrame(data)
        >>> result = convex_similarity_measure(
        ...     df, 'JaccardSimilarity', 'LevenshteinSimilarity', 'CosineSimilarity'
        ... )
        >>> print(result)
        [1.0, 0.0, 0.5]
    """
    # Create a deep copy of the DataFrame to avoid modifying the original
    df_temp = df.copy(deep=True)

    # Calculate the convex combination of similarity measures
    list_convex_normalized_similarity_measure = (
        0.25 * df_temp[similarity_measure_levenshtein_field] +
        0.25 * df_temp[similarity_measure_jaccard_field] +
        0.5 * df_temp[similarity_measure_cosine_field]
    )

    # Normalize the scores to range between 0 and 1
    list_convex_normalized_similarity_measure = (
        (list_convex_normalized_similarity_measure - list_convex_normalized_similarity_measure.min()) /
        (list_convex_normalized_similarity_measure.max() - list_convex_normalized_similarity_measure.min())
    )

    # Return the normalized scores as a list
    return list_convex_normalized_similarity_measure.tolist()


# In[20]:


def base_validation_append(df, ssn_field_name, routing_field_name,
                           aba_field_name, phone_field_name, state_field_name,
                           dl_field_name, dl_state_field_name,
                           address_benchmark_field_name, dob_field_name):
    """
    Performs a comprehensive validation of PII (Personally Identifiable Information) and bank information.

    This function applies a series of validation checks on the given fields in the DataFrame and appends
    the results as new columns to the DataFrame. It also calculates overall PII and bank validation scores.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to validate.
        ssn_field_name (str): Name of the column containing SSNs.
        routing_field_name (str): Name of the column containing routing numbers.
        aba_field_name (str): Name of the column containing account numbers.
        phone_field_name (str): Name of the column containing phone numbers.
        state_field_name (str): Name of the column containing state information.
        dl_field_name (str): Name of the column containing driver license numbers.
        dl_state_field_name (str): Name of the column containing driver license issuing states.
        address_benchmark_field_name (str): Name of the column containing address benchmark codes.
        dob_field_name (str): Name of the column containing dates of birth.

    Returns:
        pd.DataFrame: A DataFrame with appended validation results and scores.

    Example:
        >>> df = pd.DataFrame({
        ...     'SSN': ['123-45-6789', '987-65-432'],
        ...     'RoutingNumber': ['123456789', '987654321'],
        ...     'AccountNumber': ['12345', '98765'],
        ...     'MobilePhone': ['1234567890', '9876543210'],
        ...     'State': ['CA', 'NY'],
        ...     'DrivingLicenseNumber': ['A1234567', 'B9876543'],
        ...     'DrivingLicenseIssuingState': ['CA', 'NY'],
        ...     'Subject1_ReturnCode': ['A31', 'B22'],
        ...     'DateOfBirth': ['1990-01-01', None]
        ... })
        >>> result = base_validation_append(
        ...     df,
        ...     ssn_field_name='SSN',
        ...     routing_field_name='RoutingNumber',
        ...     aba_field_name='AccountNumber',
        ...     phone_field_name='MobilePhone',
        ...     state_field_name='State',
        ...     dl_field_name='DrivingLicenseNumber',
        ...     dl_state_field_name='DrivingLicenseIssuingState',
        ...     address_benchmark_field_name='Subject1_ReturnCode',
        ...     dob_field_name='DateOfBirth'
        ... )
        >>> print(result.columns)
        Index([...], dtype='object')  # Includes original and validation columns
    """
    # Create a deep copy of the DataFrame to avoid modifying the original
    df_temp = df.copy(deep=True)

    # Apply individual validation functions
    df_temp['ssn_valid'] = ssn_validation(df_temp, ssn_field_name)
    df_temp['routing_number_valid'] = routing_number_validation(
        df_temp, routing_field_name)
    df_temp['account_number_valid'] = account_bank_number_validation(
        df_temp, aba_field_name)
    df_temp['phone_valid'] = phone_validation(df_temp, phone_field_name)
    df_temp['state_valid'] = state_validation(df_temp, state_field_name)
    df_temp['dl_valid'], df_temp['dl_state_valid'] = driver_license_validation(
        df_temp, dl_field_name, dl_state_field_name)
    df_temp['address_valid'] = address_validation(
        df_temp, address_benchmark_field_name)
    df_temp['dob_valid'] = dob_validation(df_temp, dob_field_name)

    # Calculate validation scores
    df_temp['pii_validation_score'] = pii_validation_score(df_temp)
    df_temp['bank_info_validation_score'] = bank_info_validation_score(df_temp)

    # Return the DataFrame with appended validation results and scores
    return df_temp


# # Appends Validation

# ## EPS Validation

# In[21]:


def eps_validation_append(df,
                          first_name='FirstName',
                          last_name='LastName',
                          ssn_field_name='SocialSecurityNumber',
                          phone_field_name='MobilePhone',
                          city_field_name='City',
                          state_field_name='State',
                          zip_field_name='ZipCode',
                          address_field_name='streetname',
                          dob_field_name='DateOfBirth',
                          eps_first_name_1='EPS_FirstName',
                          eps_first_name_2='EPS_Alias1_FirstName',
                          eps_last_name_1='EPS_LastName',
                          eps_last_name_2='EPS_Alias1_LastName',
                          eps_ssn_field_name='EPS_SSN',
                          eps_dob_field_name='EPS_DOB',
                          eps_city_field_name='EPS_Address1_City',
                          eps_state_field_name='EPS_Address1_State',
                          eps_zip_field_name='EPS_Address1_Zip',
                          eps_address_field_name_1='EPS_Address1_Street',
                          eps_address_field_name_2='EPS_Address2_Street',
                          eps_phone_field_name_1='EPS_Phone1',
                          eps_phone_field_name_2='EPS_Phone2'):
    """
    Appends validation results and similarity measures between EPS (External Processing System) data
    and internal data in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data for validation.
        first_name (str): Column name for the first name in the internal data.
        last_name (str): Column name for the last name in the internal data.
        ssn_field_name (str): Column name for the Social Security Number in the internal data.
        phone_field_name (str): Column name for the phone number in the internal data.
        city_field_name (str): Column name for the city in the internal data.
        state_field_name (str): Column name for the state in the internal data.
        zip_field_name (str): Column name for the ZIP code in the internal data.
        address_field_name (str): Column name for the address in the internal data.
        dob_field_name (str): Column name for the date of birth in the internal data.
        eps_first_name_1, eps_first_name_2 (str): EPS first name column names.
        eps_last_name_1, eps_last_name_2 (str): EPS last name column names.
        eps_ssn_field_name (str): EPS Social Security Number column name.
        eps_dob_field_name (str): EPS date of birth column name.
        eps_city_field_name, eps_state_field_name, eps_zip_field_name (str): EPS city, state, and ZIP column names.
        eps_address_field_name_1, eps_address_field_name_2 (str): EPS address column names.
        eps_phone_field_name_1, eps_phone_field_name_2 (str): EPS phone number column names.

    Returns:
        pd.DataFrame: A DataFrame with appended validation and similarity measure columns.
    """
    df_temp = df.copy(deep=True)

    # Normalize string columns to lowercase
    for feat in [first_name,
                          last_name,
                          ssn_field_name,
                          phone_field_name,
                          city_field_name,
                          state_field_name,
                          zip_field_name,
                          address_field_name,
                          dob_field_name,
                          eps_first_name_1,
                          eps_first_name_2,
                          eps_last_name_1,
                          eps_last_name_2,
                          eps_ssn_field_name,
                          eps_dob_field_name,
                          eps_city_field_name,
                          eps_state_field_name,
                          eps_zip_field_name,
                          eps_address_field_name_1,
                          eps_address_field_name_2,
                          eps_phone_field_name_1,
                          eps_phone_field_name_2]:
        df_temp[feat] = df_temp[feat].astype(str).str.lower()

    # Basic validations
    df_temp['SSN_matched_source'] = (df_temp[ssn_field_name].astype(str) ==
                                     df_temp[eps_ssn_field_name].str[1:])
    df_temp['DOB_matched_source'] = (
        df_temp[dob_field_name] == df_temp[eps_dob_field_name])

    df_temp['EPS_city_valid'] = (
        df_temp[eps_city_field_name] == df_temp[city_field_name])
    df_temp['EPS_state_valid'] = (
        df_temp[eps_state_field_name] == df_temp[state_field_name])
    df_temp['EPS_zip_valid'] = (df_temp[eps_zip_field_name].str[:5] ==
                                df_temp[zip_field_name].astype(str).str[:5])

    # Similarity measures
    df_temp[
        'firstName_similarity_measure_jaccard_eps'] = jaccard_similarity_measure(
            df_temp, first_name, '', eps_first_name_1, eps_first_name_2)
    df_temp[
        'lastName_similarity_measure_jaccard_eps'] = jaccard_similarity_measure(
            df_temp, last_name, '', eps_last_name_1, eps_last_name_2)
    df_temp[
        'address_similarity_measure_jaccard_eps'] = jaccard_similarity_measure(
            df_temp, address_field_name, '', eps_address_field_name_1,
            eps_address_field_name_2)

    df_temp[
        'firstName_similarity_measure_levenshtein_eps'] = levenshtein_similarity_measure(
            df_temp, first_name, '', eps_first_name_1, eps_first_name_2)
    df_temp[
        'lastName_similarity_measure_levenshtein_eps'] = levenshtein_similarity_measure(
            df_temp, last_name, '', eps_last_name_1, eps_last_name_2)
    df_temp[
        'address_similarity_measure_levenshtein_eps'] = levenshtein_similarity_measure(
            df_temp, address_field_name, '', eps_address_field_name_1,
            eps_address_field_name_2)
    df_temp[
        'phone_similarity_measure_levenshtein_eps'] = levenshtein_similarity_measure(
            df_temp, phone_field_name, '', eps_phone_field_name_1,
            eps_phone_field_name_2)

    df_temp[
        'firstName_similarity_measure_cosine_eps'] = cosine_similarity_measure(
            df_temp, first_name, '', eps_first_name_1, eps_first_name_2)
    df_temp[
        'lastName_similarity_measure_cosine_eps'] = cosine_similarity_measure(
            df_temp, last_name, '', eps_last_name_1, eps_last_name_2)
    df_temp[
        'address_similarity_measure_cosine_eps'] = cosine_similarity_measure(
            df_temp, address_field_name, '', eps_address_field_name_1,
            eps_address_field_name_2)

    # Phone similarity benchmark
    df_temp['phone_similarity_measure_benchmark_eps'] = (
        df_temp['phone_similarity_measure_levenshtein_eps'] > 0.8)

    # Convex similarity measures
    df_temp[
        'firstName_similarity_convex_measure_eps'] = convex_similarity_measure(
            df_temp, 'firstName_similarity_measure_jaccard_eps',
            'firstName_similarity_measure_levenshtein_eps',
            'firstName_similarity_measure_cosine_eps')
    df_temp[
        'lastName_similarity_convex_measure_eps'] = convex_similarity_measure(
            df_temp, 'lastName_similarity_measure_jaccard_eps',
            'lastName_similarity_measure_levenshtein_eps',
            'lastName_similarity_measure_cosine_eps')
    df_temp[
        'address_similarity_convex_measure_eps'] = convex_similarity_measure(
            df_temp, 'address_similarity_measure_jaccard_eps',
            'address_similarity_measure_levenshtein_eps',
            'address_similarity_measure_cosine_eps')

    return df_temp


# ## AnchorPro Validation

# In[22]:


def anchorpro_validation_append(
        df,
        city_field_name='City',
        state_field_name='State',
        zip_field_name='ZipCode',
        address_field_name='streetname',
        anchorpro_city_field_name='Subject1_City',
        anchorpro_state_field_name='Subject1_State',
        anchorpro_zip_field_name='Subject1_Zip',
        anchorpro_address_field_name_1='Subject1_AddressLine1',
        anchorpro_address_field_name_2='Subject1_AddressLine2'):
    """
    Appends validation and similarity measures for AnchorPro address data.

    This function validates and compares address-related fields between internal data and AnchorPro data. 
    It calculates basic validations, Jaccard, Levenshtein, and Cosine similarity measures, 
    and combines them into a convex similarity score.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data for validation.
        city_field_name (str): Column name for the city in the internal data.
        state_field_name (str): Column name for the state in the internal data.
        zip_field_name (str): Column name for the ZIP code in the internal data.
        address_field_name (str): Column name for the address in the internal data.
        anchorpro_city_field_name (str): Column name for the city in AnchorPro data.
        anchorpro_state_field_name (str): Column name for the state in AnchorPro data.
        anchorpro_zip_field_name (str): Column name for the ZIP code in AnchorPro data.
        anchorpro_address_field_name_1 (str): Column name for the first address line in AnchorPro data.
        anchorpro_address_field_name_2 (str): Column name for the second address line in AnchorPro data.

    Returns:
        pd.DataFrame: A DataFrame with appended validation and similarity measure columns.
    """
    df_temp = df.copy(deep=True)

    # Normalize string columns to lowercase
    for feat in [
        city_field_name, state_field_name, zip_field_name,
        address_field_name, anchorpro_city_field_name,
        anchorpro_state_field_name, anchorpro_zip_field_name,
        anchorpro_address_field_name_1, anchorpro_address_field_name_2
    ]:
        df_temp[feat] = df_temp[feat].astype(str).str.lower()

    # Basic validations
    df_temp['anchorpro_city_valid'] = (
        df_temp[anchorpro_city_field_name] == df_temp[city_field_name]
    )
    df_temp['anchorpro_state_valid'] = (
        df_temp[anchorpro_state_field_name] == df_temp[state_field_name]
    )
    df_temp['anchorpro_zip_valid'] = (
        df_temp[anchorpro_zip_field_name].str[:5] ==
        df_temp[zip_field_name].astype(str).str[:5]
    )

    # Similarity measures
    df_temp['address_similarity_measure_jaccard_anchorpro'] = jaccard_similarity_measure(
        df_temp, address_field_name, '', anchorpro_address_field_name_1, anchorpro_address_field_name_2
    )
    df_temp['address_similarity_measure_levenshtein_anchorpro'] = levenshtein_similarity_measure(
        df_temp, address_field_name, '', anchorpro_address_field_name_1, anchorpro_address_field_name_2
    )
    df_temp['address_similarity_measure_cosine_anchorpro'] = cosine_similarity_measure(
        df_temp, address_field_name, '', anchorpro_address_field_name_1, anchorpro_address_field_name_2
    )

    # Convex similarity measures
    df_temp['address_similarity_convex_measure_anchorpro'] = convex_similarity_measure(
        df_temp, 'address_similarity_measure_jaccard_anchorpro',
        'address_similarity_measure_levenshtein_anchorpro',
        'address_similarity_measure_cosine_anchorpro'
    )

    return df_temp


# ## RVD Validation

# In[23]:


def rvd_validation_append(
        df,
        first_name='FirstName',
        last_name='LastName',
        phone_field_name='MobilePhone',
        rvd_first_name_1='firstName1',
        rvd_first_name_2='firstName2',
        rvd_last_name_1='lastName1',
        rvd_last_name_2='lastName2',
        rvd_phone_field_name_1='telephone1',
        rvd_phone_field_name_2='telephone2'):
    """
    Appends validation and similarity measures for RVD data.

    This function compares first name, last name, and phone number fields between internal data 
    and RVD data. It calculates Jaccard, Levenshtein, and Cosine similarity measures 
    and combines them into a convex similarity score.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data for validation.
        first_name (str): Column name for the first name in the internal data.
        last_name (str): Column name for the last name in the internal data.
        phone_field_name (str): Column name for the phone number in the internal data.
        rvd_first_name_1, rvd_first_name_2 (str): RVD first name column names.
        rvd_last_name_1, rvd_last_name_2 (str): RVD last name column names.
        rvd_phone_field_name_1, rvd_phone_field_name_2 (str): RVD phone number column names.

    Returns:
        pd.DataFrame: A DataFrame with appended validation and similarity measure columns.
    """
    df_temp = df.copy(deep=True)

    # Normalize string columns to lowercase
    for feat in [
        first_name, last_name, phone_field_name,
        rvd_first_name_1, rvd_first_name_2,
        rvd_last_name_1, rvd_last_name_2,
        rvd_phone_field_name_1, rvd_phone_field_name_2
    ]:
        df_temp[feat] = df_temp[feat].astype(str).str.lower()

    # Similarity measures
    df_temp['firstName_similarity_measure_jaccard_rvd'] = jaccard_similarity_measure(
        df_temp, first_name, '', rvd_first_name_1, rvd_first_name_2
    )
    df_temp['lastName_similarity_measure_jaccard_rvd'] = jaccard_similarity_measure(
        df_temp, last_name, '', rvd_last_name_1, rvd_last_name_2
    )

    df_temp['firstName_similarity_measure_levenshtein_rvd'] = levenshtein_similarity_measure(
        df_temp, first_name, '', rvd_first_name_1, rvd_first_name_2
    )
    df_temp['lastName_similarity_measure_levenshtein_rvd'] = levenshtein_similarity_measure(
        df_temp, last_name, '', rvd_last_name_1, rvd_last_name_2
    )
    df_temp['phone_similarity_measure_levenshtein_rvd'] = levenshtein_similarity_measure(
        df_temp, phone_field_name, '', rvd_phone_field_name_1, rvd_phone_field_name_2
    )

    df_temp['firstName_similarity_measure_cosine_rvd'] = cosine_similarity_measure(
        df_temp, first_name, '', rvd_first_name_1, rvd_first_name_2
    )
    df_temp['lastName_similarity_measure_cosine_rvd'] = cosine_similarity_measure(
        df_temp, last_name, '', rvd_last_name_1, rvd_last_name_2
    )

    # Phone similarity benchmark
    df_temp['phone_similarity_measure_benchmark_rvd'] = (
        df_temp['phone_similarity_measure_levenshtein_rvd'] > 0.8
    )

    # Convex similarity measures
    df_temp['firstName_similarity_convex_measure_rvd'] = convex_similarity_measure(
        df_temp, 'firstName_similarity_measure_jaccard_rvd',
        'firstName_similarity_measure_levenshtein_rvd',
        'firstName_similarity_measure_cosine_rvd'
    )
    df_temp['lastName_similarity_convex_measure_rvd'] = convex_similarity_measure(
        df_temp, 'lastName_similarity_measure_jaccard_rvd',
        'lastName_similarity_measure_levenshtein_rvd',
        'lastName_similarity_measure_cosine_rvd'
    )

    return df_temp


# # Main Part

# In[26]:


df=pd.read_csv("DogstarF1_outint_by_mbid_new.csv")
df_anchor=pd.read_csv("AnchorPro _DogstarF1_20241015_INPUT_output.csv")
df_superphone=pd.read_csv("SuperPhone_DogstarF1_20241015_INPUT_output.csv")
df_eps=pd.read_csv("EPS_DogstarF1_20241015_INPUT_output.csv")
df_res=pd.concat([pd.concat([pd.concat([df, df_eps], axis=1), df_anchor], axis=1), df_superphone], axis=1)


# In[28]:


df = base_validation_append(df_res,
                            ssn_field_name='SocialSecurityNumber',
                            routing_field_name='RoutingNumber',
                            aba_field_name='AccountNumber',
                            phone_field_name='MobilePhone',
                            state_field_name='State',
                            dl_field_name='DrivingLicenseNumber',
                            dl_state_field_name='DrivingLicenseIssuingState',
                            address_benchmark_field_name='Subject1_ReturnCode',
                            dob_field_name='DateOfBirth')



df = eps_validation_append(df)
df = anchorpro_validation_append(df)
df = rvd_validation_append(df)

