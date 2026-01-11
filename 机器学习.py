# ============================================================================
# Part 1: Import necessary libraries
# ============================================================================

# Basic data processing libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Model evaluation
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report,
                             precision_recall_curve)

# Other tools
import time
import joblib
import json
import os
from datetime import datetime
from collections import Counter

# Set Chinese display (if needed)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")

print("✅ All necessary libraries imported")


# ============================================================================
# Part 2: Data loading and initial exploration
# ============================================================================

def load_and_explore_data(train_path='train.csv', test_path='test.csv'):
    """
    Load data and perform initial exploration
    """
    print("📊 Loading data...")

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"✅ Data loading completed!")
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    # Display basic information
    print("\n" + "=" * 60)
    print("Training set basic information:")
    print("=" * 60)
    print(train_df.info())

    print("\n" + "=" * 60)
    print("First 5 rows of training set:")
    print("=" * 60)
    print(train_df.head())

    print("\n" + "=" * 60)
    print("Training set descriptive statistics:")
    print("=" * 60)
    print(train_df.describe())

    # Check target variable
    print("\n" + "=" * 60)
    print("Target variable 'Transported' distribution:")
    print("=" * 60)
    transported_counts = train_df['Transported'].value_counts()
    transported_percent = train_df['Transported'].value_counts(normalize=True) * 100

    for status, (count, percent) in enumerate(zip(transported_counts, transported_percent)):
        label = "Transported" if status else "Not Transported"
        print(f"{label}: {count} ({percent:.2f}%)")

    return train_df, test_df


# Execute data loading (assuming data files are in current directory)
train_data, test_data = load_and_explore_data()

# Save copies of original data
train_original = train_data.copy()
test_original = test_data.copy()


# ============================================================================
# Part 3: Exploratory Data Analysis (EDA)
# ============================================================================

def perform_eda(train_df, test_df):
    """
    Perform comprehensive exploratory data analysis
    """
    print("🔍 Starting exploratory data analysis...")

    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)

    # 3.1 Missing value analysis
    print("\n" + "=" * 60)
    print("Missing value analysis")
    print("=" * 60)

    def analyze_missing_data(df, dataset_name):
        """Analyze missing data"""
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100

        missing_df = pd.DataFrame({
            'Feature': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_percent.values
        }).sort_values('Missing_Percentage', ascending=False)

        missing_df = missing_df[missing_df['Missing_Count'] > 0]

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Bar chart
        if len(missing_df) > 0:
            axes[0].barh(missing_df['Feature'], missing_df['Missing_Count'])
            axes[0].set_xlabel('Number of Missing Values')
            axes[0].set_title(f'{dataset_name} - Number of Missing Values')

            axes[1].barh(missing_df['Feature'], missing_df['Missing_Percentage'])
            axes[1].set_xlabel('Missing Percentage (%)')
            axes[1].set_title(f'{dataset_name} - Missing Percentage')

            # Add value labels
            for ax in axes:
                for bar in ax.patches:
                    width = bar.get_width()
                    ax.text(width + max(width * 0.01, 0.5),
                            bar.get_y() + bar.get_height() / 2,
                            f'{width:.1f}',
                            ha='left', va='center', fontsize=9)
        else:
            axes[0].text(0.5, 0.5, 'No missing values',
                         ha='center', va='center',
                         transform=axes[0].transAxes, fontsize=12)
            axes[0].axis('off')
            axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(f'visualizations/{dataset_name}_missing_values.png', dpi=300, bbox_inches='tight')
        plt.show()

        return missing_df

    train_missing = analyze_missing_data(train_df, 'Training_Set')
    test_missing = analyze_missing_data(test_df, 'Test_Set')

    # 3.2 Target variable analysis
    print("\n" + "=" * 60)
    print("Target variable analysis")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    transported_counts = train_df['Transported'].value_counts()
    labels = ['Not Transported', 'Transported']
    colors = ['#FF6B6B', '#4ECDC4']

    axes[0].pie(transported_counts, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05))
    axes[0].set_title('Target Variable Distribution (Transported)')

    # Bar chart
    bars = axes[1].bar(labels, transported_counts.values, color=colors)
    axes[1].set_title('Target Variable Count Distribution')
    axes[1].set_ylabel('Count')

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height + 20,
                     f'{height}\n({height / len(train_df) * 100:.1f}%)',
                     ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('visualizations/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3.3 Numeric feature analysis
    print("\n" + "=" * 60)
    print("Numeric feature analysis")
    print("=" * 60)

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Transported' in numeric_cols:
        numeric_cols.remove('Transported')

    if numeric_cols:
        n_cols = min(6, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten()

        for idx, col in enumerate(numeric_cols):
            if idx < len(axes):
                # Histogram
                axes[idx].hist(train_df[col].dropna(), bins=30, alpha=0.7,
                               color='skyblue', edgecolor='black')
                axes[idx].set_title(f'{col}', fontsize=10)
                axes[idx].set_xlabel(col, fontsize=8)
                axes[idx].set_ylabel('Frequency', fontsize=8)
                axes[idx].tick_params(labelsize=8)

                # Add statistical information
                mean_val = train_df[col].mean()
                median_val = train_df[col].median()
                axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.2f}')
                axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=1,
                                  label=f'Median: {median_val:.2f}')
                axes[idx].legend(fontsize=7)

        # Hide extra subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('visualizations/numeric_features_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 3.4 Categorical feature analysis
    print("\n" + "=" * 60)
    print("Categorical feature analysis")
    print("=" * 60)

    categorical_cols = train_df.select_dtypes(include=['object', 'bool']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != 'Transported']

    if categorical_cols:
        n_cols = min(4, len(categorical_cols))
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()

        for idx, col in enumerate(categorical_cols):
            if idx < len(axes):
                # Calculate category frequency
                value_counts = train_df[col].value_counts().head(10)  # Show only top 10

                # Bar chart
                bars = axes[idx].bar(range(len(value_counts)), value_counts.values)
                axes[idx].set_title(f'{col} Distribution', fontsize=10)
                axes[idx].set_xlabel(col, fontsize=8)
                axes[idx].set_ylabel('Frequency', fontsize=8)
                axes[idx].set_xticks(range(len(value_counts)))
                axes[idx].set_xticklabels([str(x)[:10] for x in value_counts.index],
                                          rotation=45, fontsize=8, ha='right')

                # Add values on top of bars
                for bar, count in zip(bars, value_counts.values):
                    height = bar.get_height()
                    axes[idx].text(bar.get_x() + bar.get_width() / 2., height + max(height * 0.01, 1),
                                   str(count), ha='center', va='bottom', fontsize=7)

        # Hide extra subplots
        for idx in range(len(categorical_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('visualizations/categorical_features_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 3.5 Relationship analysis between features and target variable
    print("\n" + "=" * 60)
    print("Relationship analysis between features and target variable")
    print("=" * 60)

    # Select several important features for analysis
    important_features = ['CryoSleep', 'VIP', 'HomePlanet', 'Destination', 'Age']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, col in enumerate(important_features):
        if idx < len(axes) and col in train_df.columns:
            if col in numeric_cols:
                # Numeric feature: box plot
                data_to_plot = []
                labels = []
                for transported in [False, True]:
                    data = train_df[train_df['Transported'] == transported][col].dropna()
                    if len(data) > 0:
                        data_to_plot.append(data)
                        labels.append('Transported' if transported else 'Not Transported')

                axes[idx].boxplot(data_to_plot, labels=labels)
                axes[idx].set_title(f'{col} vs Transported', fontsize=10)
                axes[idx].set_ylabel(col, fontsize=8)

            else:
                # Categorical feature: stacked bar chart
                cross_tab = pd.crosstab(train_df[col], train_df['Transported'], normalize='index') * 100

                if not cross_tab.empty:
                    bottom = np.zeros(len(cross_tab))
                    colors = ['#FF6B6B', '#4ECDC4']

                    for i, transported in enumerate([False, True]):
                        values = cross_tab[transported].values
                        axes[idx].bar(range(len(cross_tab)), values, bottom=bottom,
                                      color=colors[i], label='Transported' if transported else 'Not Transported')
                        bottom += values

                    axes[idx].set_title(f'{col} vs Transported', fontsize=10)
                    axes[idx].set_xlabel(col, fontsize=8)
                    axes[idx].set_ylabel('Percentage (%)', fontsize=8)
                    axes[idx].set_xticks(range(len(cross_tab)))
                    axes[idx].set_xticklabels([str(x)[:15] for x in cross_tab.index],
                                              rotation=45, fontsize=8, ha='right')
                    axes[idx].legend(fontsize=8)

    # Hide extra subplots
    for idx in range(len(important_features), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('visualizations/feature_target_relationship.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3.6 Spending feature analysis
    print("\n" + "=" * 60)
    print("Spending feature analysis")
    print("=" * 60)

    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Individual spending features
    for idx, col in enumerate(spending_cols):
        if idx < len(axes) and col in train_df.columns:
            # Histogram after log transformation
            data = train_df[col].dropna()
            log_data = np.log1p(data)  # log(1+x)

            axes[idx].hist(log_data, bins=30, alpha=0.7, color='purple', edgecolor='black')
            axes[idx].set_title(f'log(1+{col}) Distribution', fontsize=10)
            axes[idx].set_xlabel(f'log(1+{col})', fontsize=8)
            axes[idx].set_ylabel('Frequency', fontsize=8)

    # Total spending
    if len(spending_cols) > 0:
        total_spending = train_df[spending_cols].sum(axis=1, skipna=True)
        log_total = np.log1p(total_spending)

        axes[5].hist(log_total.dropna(), bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[5].set_title('log(1+Total Spending) Distribution', fontsize=10)
        axes[5].set_xlabel('log(1+Total Spending)', fontsize=8)
        axes[5].set_ylabel('Frequency', fontsize=8)

    plt.tight_layout()
    plt.savefig('visualizations/spending_features_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Exploratory data analysis completed!")

    return {
        'train_missing': train_missing,
        'test_missing': test_missing,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'spending_cols': spending_cols
    }


# Execute EDA
eda_results = perform_eda(train_data, test_data)


# ============================================================================
# Part 4: Feature Engineering
# ============================================================================

def feature_engineering(df, is_train=True):
    """
    Perform feature engineering: create new features, process existing features
    """
    print("🔧 Starting feature engineering...")

    df = df.copy()

    # 4.1 Extract information from PassengerId
    print("  Extracting group information from PassengerId...")
    df['GroupId'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
    df['PersonId'] = df['PassengerId'].apply(lambda x: x.split('_')[1]).astype(int)

    # Calculate group size and position within group
    df['GroupSize'] = df.groupby('GroupId')['GroupId'].transform('count')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    df['IsFirstInGroup'] = (df['PersonId'] == 1).astype(int)
    df['IsLastInGroup'] = (df['PersonId'] == df['GroupSize']).astype(int)

    # 4.2 Process Cabin feature
    print("  Processing Cabin feature...")

    def extract_cabin_info(cabin):
        if pd.isna(cabin):
            return {'Deck': 'Unknown', 'Num': -1, 'Side': 'Unknown'}

        try:
            parts = str(cabin).split('/')
            if len(parts) == 3:
                return {
                    'Deck': parts[0],
                    'Num': int(parts[1]) if parts[1].isdigit() else -1,
                    'Side': parts[2]
                }
        except:
            pass

        return {'Deck': 'Unknown', 'Num': -1, 'Side': 'Unknown'}

    cabin_info = df['Cabin'].apply(extract_cabin_info)
    df['Deck'] = cabin_info.apply(lambda x: x['Deck'])
    df['CabinNum'] = cabin_info.apply(lambda x: x['Num'])
    df['Side'] = cabin_info.apply(lambda x: x['Side'])

    # Deck grouping (based on data analysis)
    deck_groups = {
        'A': 'Group_A',
        'B': 'Group_AB',
        'C': 'Group_AB',
        'D': 'Group_CD',
        'E': 'Group_CD',
        'F': 'Group_FG',
        'G': 'Group_FG',
        'T': 'Group_T',
        'Unknown': 'Unknown'
    }
    df['DeckGroup'] = df['Deck'].map(deck_groups)

    # 4.3 Process Name feature
    print("  Processing Name feature...")
    df['Surname'] = df['Name'].apply(lambda x: str(x).split()[-1] if pd.notna(x) else 'Unknown')
    df['NameLength'] = df['Name'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

    # Create family size feature
    df['FamilySize'] = df.groupby(['GroupId', 'Surname'])['Surname'].transform('count')
    df['HasFamily'] = (df['FamilySize'] > 1).astype(int)

    # 4.4 Spending feature engineering
    print("  Processing spending features...")
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Total spending
    df['TotalSpending'] = df[spending_cols].sum(axis=1, skipna=True)

    # Has any spending
    df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)

    # Spending diversity (number of services with spending)
    df['SpendingDiversity'] = (df[spending_cols] > 0).sum(axis=1)

    # Average spending (if any spending)
    df['AvgSpending'] = df[spending_cols].mean(axis=1, skipna=True)

    # Max spending
    df['MaxSpending'] = df[spending_cols].max(axis=1, skipna=True)

    # Spending ratio features
    for col in spending_cols:
        df[f'{col}_Ratio'] = df[col] / (df['TotalSpending'] + 1)  # +1 to avoid division by zero

    # Log transformation (to handle skewness)
    for col in spending_cols + ['TotalSpending']:
        df[f'log_{col}'] = np.log1p(df[col].fillna(0))

    # 4.5 Age feature engineering
    print("  Processing age features...")
    # Age grouping
    age_bins = [0, 12, 18, 30, 50, 100]
    age_labels = ['Child', 'Teen', 'YoungAdult', 'MiddleAge', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

    # Boolean features
    df['IsChild'] = (df['Age'] < 13).astype(int)
    df['IsTeen'] = ((df['Age'] >= 13) & (df['Age'] < 19)).astype(int)
    df['IsAdult'] = (df['Age'] >= 19).astype(int)
    df['IsElderly'] = (df['Age'] > 50).astype(int)

    # 4.6 Interaction features
    print("  Creating interaction features...")
    # Age and spending interaction
    df['Age_TotalSpending'] = df['Age'] * df['TotalSpending']

    # CryoSleep and spending interaction
    if 'CryoSleep' in df.columns:
        df['CryoSleep_NoSpending'] = (df['CryoSleep'] & (df['TotalSpending'] == 0)).astype(int)

    # VIP and spending interaction
    if 'VIP' in df.columns:
        df['VIP_HighSpending'] = (df['VIP'] & (df['TotalSpending'] > df['TotalSpending'].median())).astype(int)

    # 4.7 Cabin number features
    print("  Processing cabin number features...")
    # Cabin number parity (may affect transportation)
    df['CabinEven'] = (df['CabinNum'] % 2 == 0).astype(int)
    df['CabinEven'] = df['CabinEven'].where(df['CabinNum'] != -1, -1)

    # Cabin number range
    # First create a temporary Series
    cabin_sections = pd.cut(df['CabinNum'],
                            bins=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, float('inf')],
                            labels=['0-100', '100-200', '200-300', '300-400', '400-500',
                                    '500-600', '600-700', '700-800', '800-900', '900-1000', '1000+'])

    # Convert to string type
    df['CabinSection'] = cabin_sections.astype(str)

    # Set NaN and -1 cases to 'Unknown'
    df['CabinSection'] = df['CabinSection'].where(df['CabinNum'] != -1, 'Unknown')
    df['CabinSection'] = df['CabinSection'].replace('nan', 'Unknown')

    # 4.8 Create derived features
    print("  Creating derived features...")
    # Family spending per person
    df['FamilySpendingPerPerson'] = df['TotalSpending'] / df['FamilySize']
    df['FamilySpendingPerPerson'] = df['FamilySpendingPerPerson'].replace([np.inf, -np.inf], 0)

    # Spending rank within group
    df['SpendingRankInGroup'] = df.groupby('GroupId')['TotalSpending'].rank(method='dense', ascending=False)

    # Age rank within group
    df['AgeRankInGroup'] = df.groupby('GroupId')['Age'].rank(method='dense', ascending=False)

    print(f"✅ Feature engineering completed! Created {len(df.columns) - len(df.columns)} new features")
    print(f"   Total features: {df.shape[1]}")

    return df


# Apply feature engineering
print("Applying feature engineering to training set...")
train_fe = feature_engineering(train_data, is_train=True)

print("\nApplying feature engineering to test set...")
test_fe = feature_engineering(test_data, is_train=False)

# Display new features
print("\n" + "=" * 60)
print("Newly created features (first 20):")
print("=" * 60)
new_features = [col for col in train_fe.columns if col not in train_data.columns]
print(f"Total new features created: {len(new_features)}")
print(f"New features list: {new_features[:20]}...")


# ============================================================================
# Part 5: Data Preprocessing and Missing Value Handling
# ============================================================================

def preprocess_data(train_df, test_df):
    """
    Data preprocessing: handle missing values, encode categorical variables, etc.
    """
    print("🔄 Starting data preprocessing...")

    train_processed = train_df.copy()
    test_processed = test_df.copy()

    # 5.1 Handle missing values
    print("  Handling missing values...")

    # Numeric features filled with median
    numeric_cols = train_processed.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Transported']

    for col in numeric_cols:
        if col in train_processed.columns:
            median_val = train_processed[col].median()
            train_processed[col] = train_processed[col].fillna(median_val)
            if col in test_processed.columns:
                test_processed[col] = test_processed[col].fillna(median_val)

    # Spending features: missing values treated as 0 (no spending)
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in spending_cols:
        if col in train_processed.columns:
            train_processed[col] = train_processed[col].fillna(0)
            if col in test_processed.columns:
                test_processed[col] = test_processed[col].fillna(0)

    # Categorical features filled with mode or create "Missing" category
    categorical_cols = train_processed.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_cols:
        if col in train_processed.columns:
            # Training set
            if train_processed[col].isnull().any():
                # If many missing values, create "Missing" category
                missing_percent = train_processed[col].isnull().mean()
                if missing_percent > 0.1:  # More than 10% missing
                    train_processed[col] = train_processed[col].fillna('Missing')
                else:
                    mode_val = train_processed[col].mode()[0] if not train_processed[col].mode().empty else 'Unknown'
                    train_processed[col] = train_processed[col].fillna(mode_val)

            # Test set
            if col in test_processed.columns:
                if test_processed[col].isnull().any():
                    # Fill test set with training set mode
                    mode_val = train_processed[col].mode()[0] if not train_processed[col].mode().empty else 'Unknown'
                    test_processed[col] = test_processed[col].fillna(mode_val)

    # Boolean features
    bool_cols = train_processed.select_dtypes(include=['bool']).columns.tolist()
    bool_cols = [col for col in bool_cols if col != 'Transported']

    for col in bool_cols:
        if col in train_processed.columns:
            mode_val = train_processed[col].mode()[0] if not train_processed[col].mode().empty else False
            train_processed[col] = train_processed[col].fillna(mode_val)
            if col in test_processed.columns:
                test_processed[col] = test_processed[col].fillna(mode_val)

    # 5.2 Check if there are still missing values
    print("\n  Checking missing value handling results...")
    train_missing = train_processed.isnull().sum().sum()
    test_missing = test_processed.isnull().sum().sum()

    print(f"  Remaining missing values in training set: {train_missing}")
    print(f"  Remaining missing values in test set: {test_missing}")

    if train_missing > 0 or test_missing > 0:
        print("  ⚠️ Warning: There are still missing values!")

        # Show which columns still have missing values
        missing_cols_train = train_processed.columns[train_processed.isnull().any()].tolist()
        missing_cols_test = test_processed.columns[test_processed.isnull().any()].tolist()

        if missing_cols_train:
            print(f"  Columns with missing values in training set: {missing_cols_train}")
        if missing_cols_test:
            print(f"  Columns with missing values in test set: {missing_cols_test}")

        # Fill remaining missing values with simple method
        for df in [train_processed, test_processed]:
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in [np.float64, np.int64]:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna('Unknown')

    # 5.3 Save processed data
    train_processed.to_csv('train_processed.csv', index=False)
    test_processed.to_csv('test_processed.csv', index=False)

    print("\n✅ Data preprocessing completed!")
    print(f"  Training set shape: {train_processed.shape}")
    print(f"  Test set shape: {test_processed.shape}")
    print("  Processed data saved as 'train_processed.csv' and 'test_processed.csv'")

    return train_processed, test_processed


# Execute data preprocessing
train_processed, test_processed = preprocess_data(train_fe, test_fe)


# ============================================================================
# Part 6: Feature Encoding and Selection
# ============================================================================

def encode_and_select_features(train_df, test_df):
    """
    Encode categorical features and perform feature selection
    """
    print("🎯 Starting feature encoding and selection...")

    # Separate target variable
    if 'Transported' in train_df.columns:
        y = train_df['Transported'].astype(int)
        X_train = train_df.drop('Transported', axis=1)
    else:
        raise ValueError("'Transported' column not found in training set")

    X_test = test_df.copy()

    # 6.1 Drop unnecessary columns
    print("  Dropping unnecessary columns...")
    cols_to_drop = ['PassengerId', 'Name', 'Cabin', 'Surname']
    for col in cols_to_drop:
        if col in X_train.columns:
            X_train = X_train.drop(col, axis=1)
        if col in X_test.columns:
            X_test = X_test.drop(col, axis=1)

    # 6.2 Encode categorical variables
    print("  Encoding categorical variables...")

    # Identify categorical features
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Encode using LabelEncoder
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()

        # Combine categories from training and test sets
        train_values = X_train[col].astype(str).unique()
        test_values = X_test[col].astype(str).unique() if col in X_test.columns else []
        all_values = np.unique(np.concatenate([train_values, test_values]))

        # Train encoder
        le.fit(all_values)

        # Transform training set
        X_train[col] = le.transform(X_train[col].astype(str))

        # Transform test set
        if col in X_test.columns:
            X_test[col] = le.transform(X_test[col].astype(str))

        label_encoders[col] = le

    # 6.3 Feature selection
    print("  Performing feature selection...")

    # Method 1: Feature selection based on ANOVA
    selector_anova = SelectKBest(score_func=f_classif, k='all')
    selector_anova.fit(X_train, y)

    # Method 2: Feature selection based on mutual information
    selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
    selector_mi.fit(X_train, y)

    # Create feature importance DataFrame
    feature_scores = pd.DataFrame({
        'Feature': X_train.columns,
        'ANOVA_Score': selector_anova.scores_,
        'ANOVA_pvalue': selector_anova.pvalues_,
        'MI_Score': selector_mi.scores_
    })

    # Normalize scores
    feature_scores['ANOVA_Score_Norm'] = (feature_scores['ANOVA_Score'] - feature_scores['ANOVA_Score'].min()) / \
                                         (feature_scores['ANOVA_Score'].max() - feature_scores[
                                             'ANOVA_Score'].min() + 1e-10)

    feature_scores['MI_Score_Norm'] = (feature_scores['MI_Score'] - feature_scores['MI_Score'].min()) / \
                                      (feature_scores['MI_Score'].max() - feature_scores['MI_Score'].min() + 1e-10)

    # Combined score
    feature_scores['Combined_Score'] = 0.7 * feature_scores['ANOVA_Score_Norm'] + 0.3 * feature_scores['MI_Score_Norm']

    # Sort
    feature_scores = feature_scores.sort_values('Combined_Score', ascending=False)

    # Select top k features
    k = min(30, len(feature_scores))  # Select top 30 best features
    selected_features = feature_scores.head(k)['Feature'].tolist()

    print(f"  Selected {k} best features")

    # 6.4 Visualize feature importance
    print("  Visualizing feature importance...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Top 20 features
    top_20 = feature_scores.head(20)

    # Combined score bar chart
    bars = axes[0].barh(range(len(top_20)), top_20['Combined_Score'])
    axes[0].set_yticks(range(len(top_20)))
    axes[0].set_yticklabels(top_20['Feature'], fontsize=9)
    axes[0].set_xlabel('Combined Importance Score')
    axes[0].set_title('Top 20 Feature Importance (Combined Score)')
    axes[0].invert_yaxis()

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top_20['Combined_Score'])):
        axes[0].text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=8)

    # Scatter plot: ANOVA vs MI
    axes[1].scatter(feature_scores['ANOVA_Score_Norm'],
                    feature_scores['MI_Score_Norm'],
                    alpha=0.6)

    # Mark selected features
    for idx, row in feature_scores.iterrows():
        if row['Feature'] in selected_features:
            axes[1].scatter(row['ANOVA_Score_Norm'], row['MI_Score_Norm'],
                            color='red', alpha=0.8, s=50)
            axes[1].annotate(row['Feature'][:15],
                             (row['ANOVA_Score_Norm'], row['MI_Score_Norm']),
                             fontsize=8, alpha=0.7)

    axes[1].set_xlabel('ANOVA Score (Normalized)')
    axes[1].set_ylabel('Mutual Information Score (Normalized)')
    axes[1].set_title('Feature Selection Scatter Plot (Red = Selected Features)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 6.5 Keep only selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # 6.6 Feature standardization
    print("  Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Convert to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=selected_features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=selected_features)

    print("\n✅ Feature encoding and selection completed!")
    print(f"  Original number of features: {len(train_df.columns) - 1}")
    print(f"  Number of selected features: {k}")
    print(f"  Training set shape: {X_train_scaled_df.shape}")
    print(f"  Test set shape: {X_test_scaled_df.shape}")

    # Save selected feature names
    with open('selected_features.json', 'w') as f:
        json.dump(selected_features, f)

    return (X_train_scaled_df, y, X_test_scaled_df, selected_features,
            label_encoders, scaler, feature_scores)


# Execute feature encoding and selection
X_train_scaled, y_train, X_test_scaled, selected_features, label_encoders, scaler, feature_scores = encode_and_select_features(
    train_processed, test_processed
)


# ============================================================================
# Part 7: Model Training and Evaluation
# ============================================================================

def train_and_evaluate_models(X_train, y_train, X_test=None, y_test=None):
    """
    Train multiple models and perform evaluation
    """
    print("🤖 Starting model training and evaluation...")

    # 7.1 Split training and validation sets
    print("  Splitting training and validation sets...")
    if X_test is None or y_test is None:
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    else:
        X_train_split, X_val, y_train_split, y_val = X_train, X_test, y_train, y_test

    print(f"  Training set size: {X_train_split.shape}")
    print(f"  Validation set size: {X_val.shape}")

    # 7.2 Define models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=0.1),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42, learning_rate=0.1),
        'SVM': SVC(probability=True, random_state=42, kernel='rbf', C=1.0),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, max_depth=5,
                                 learning_rate=0.1, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, max_depth=5,
                                   learning_rate=0.1, verbose=-1)
    }

    # 7.3 Train and evaluate models
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n  🚀 Training {name}...")
        start_time = time.time()

        try:
            # Train model
            model.fit(X_train_split, y_train_split)
            training_time = time.time() - start_time

            # Predict
            y_pred_train = model.predict(X_train_split)
            y_pred_val = model.predict(X_val)

            # Predict probabilities (if supported)
            if hasattr(model, 'predict_proba'):
                y_prob_val = model.predict_proba(X_val)[:, 1]
                auc_score = roc_auc_score(y_val, y_prob_val)
            else:
                y_prob_val = None
                auc_score = None

            # Calculate metrics
            metrics = {
                'Model': name,
                'Training_Time': training_time,
                'Training_Accuracy': accuracy_score(y_train_split, y_pred_train),
                'Validation_Accuracy': accuracy_score(y_val, y_pred_val),
                'Validation_Precision': precision_score(y_val, y_pred_val),
                'Validation_Recall': recall_score(y_val, y_pred_val),