#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==============================================================================
# Mental Health Comorbidity Analysis Pipeline
# MH-CLD 2021: Latent Class Analysis & Network Modeling
# ==============================================================================
"""
Comprehensive analytical pipeline for mental health comorbidity patterns.
Integrates Latent Class Analysis (LCA) with network modeling to identify
transdiagnostic subtypes in the 2021 Mental Health Client-Level Data.

Author: Mounika Supriya Bommi
Institution: George Mason University
Contact: mbommi@gmu.edu

Dependencies: numpy, pandas, matplotlib, scikit-learn, networkx, scipy
"""

# ==============================================================================
# 1. IMPORT LIBRARIES & CONFIGURATION
# ==============================================================================

import os
import json
import gc
import logging
import warnings
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server deployment
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from joblib import Parallel, delayed
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# Configure logging and warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("mh_comorbidity_analysis")

# ==============================================================================
# 2. VISUALIZATION SETTINGS & CONSTANTS
# ==============================================================================

# Publication-quality figure settings
FIGURE_DPI = 300
SAVE_PNG_FORMAT = True
SAVE_PDF_FORMAT = True
DEFAULT_FILE_EXTENSION = "png"

# Journal-style formatting
plt.style.use("seaborn-v0_8-pastel")
plt.rcParams.update({
    "figure.figsize": (7.5, 5.0),
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

# Accessible color palettes for scientific publication
QUALITATIVE_PALETTE = ["#cecece", "#a559aa", "#59a89c", "#f0c571", "#e02b35", "#082a54"]
DIVERGING_PALETTE = ["#2066a8", "#8ec1da", "#cde1ec", "#ededed", "#f6d6c2", "#d47264", "#ae282c"]
SEQUENTIAL_PALETTE = ["#b5d1ae", "#80ae9a", "#568b87", "#326b77", "#1b485e", "#122740"]

# ==============================================================================
# 3. COLOR UTILITY FUNCTIONS
# ==============================================================================

def get_color_palette(n_colors: int, palette: Optional[List[str]] = None) -> List[str]:
    """
    Generate a color palette with specified number of colors.
    
    Args:
        n_colors: Number of distinct colors required
        palette: Base color palette (defaults to qualitative palette)
    
    Returns:
        List of hex color codes
    """
    base_palette = palette or QUALITATIVE_PALETTE
    if n_colors <= len(base_palette):
        return base_palette[:n_colors]
    repetitions = math.ceil(n_colors / len(base_palette))
    return (base_palette * repetitions)[:n_colors]


def darken(hex_color: str, darkening_factor: float = 0.85) -> str:
    """
    Create a darker variant of a hex color for visual hierarchy.
    
    Args:
        hex_color: Original color in hex format
        darkening_factor: Multiplier for RGB values (0-1)
    
    Returns:
        Darkened hex color
    """
    rgb_color = matplotlib.colors.to_rgb(hex_color)
    darkened_rgb = tuple(max(0.0, channel * darkening_factor) for channel in rgb_color)
    return to_hex(darkened_rgb)

# ==============================================================================
# 4. ANALYSIS CONFIGURATION
# ==============================================================================

@dataclass
class AnalysisConfiguration:
    """Central configuration for the comorbidity analysis pipeline."""
    
    # Data paths
    DATA_FILE_PATH: str = os.environ.get(
        "MHCLD_DATA_PATH",
        r"C:\path\to\mhcld_puf_2021.csv"
    )
    OUTPUT_DIRECTORY: str = os.environ.get("MHCLD_OUTPUT_DIR", "./publication_outputs")
    RANDOM_SEED: int = 42

    # Sampling parameters
    NUM_DIAGNOSES_KEEP: tuple = (1, 2)
    SAMPLE_SIZE_PER_GROUP: int = 50_000
    USE_POSTSTRATIFICATION_WEIGHTS: bool = True

    # Data quality parameters
    MISSING_DATA_CODE: int = -9
    MISSING_DATA_THRESHOLD: float = 0.60
    IMPUTATION_CHUNK_SIZE: int = 100_000

    # Latent Class Analysis parameters
    CLASS_RANGE: range = range(2, 7)
    NUMBER_INITIALIZATIONS: int = 8
    MAXIMUM_ITERATIONS: int = 500
    CONVERGENCE_TOLERANCE: float = 1e-6

    # Classification parameters
    CLASSIFICATION_CONFIDENCE_THRESHOLD: float = 0.70

    # Network analysis parameters
    MINIMUM_EDGE_SUPPORT: int = 25
    NORMALIZE_NETWORK_WEIGHTS: bool = True

    # Statistical validation parameters
    BOOTSTRAP_ITERATIONS: int = 50
    PARALLEL_JOBS: int = -1

    # Diagnostic controls
    ENABLE_MODEL_DIAGNOSTICS: bool = True
    ENABLE_SENSITIVITY_ANALYSIS: bool = True
    DATA_PERTURBATION_RATE: float = 0.02
    NETWORK_THRESHOLDS: tuple = (10, 25, 50)

    # Data column specifications
    DIAGNOSIS_FLAG_COLUMNS: List[str] = field(default_factory=lambda: [
        "TRAUSTREFLG", "ANXIETYFLG", "ADHDFLG", "CONDUCTFLG", "DELIRDEMFLG",
        "BIPOLARFLG", "DEPRESSFLG", "ODDFLG", "PDDFLG", "PERSONFLG", "SCHIZOFLG",
        "ALCSUBFLG", "OTHERDISFLG"
    ])
    
    DEMOGRAPHIC_COLUMNS: List[str] = field(default_factory=lambda: [
        "AGE_GROUP", "GENDER", "SMISED", "RACE", "EDUC", "MARSTAT", "SAP", "ETHNIC"
    ])
    
    DSM5_CLUSTER_DEFINITIONS: Dict[str, List[str]] = field(default_factory=lambda: {
        'Neurodevelopmental': ['ADHDFLG', 'PDDFLG'],
        'Schizophrenia_Spectrum': ['SCHIZOFLG'],
        'Bipolar_Related': ['BIPOLARFLG'],
        'Depressive_Disorders': ['DEPRESSFLG'],
        'Anxiety_Disorders': ['ANXIETYFLG'],
        'Trauma_Stressor_Related': ['TRAUSTREFLG'],
        'Disruptive_Impulse_Control': ['CONDUCTFLG', 'ODDFLG'],
        'Substance_Related': ['ALCSUBFLG'],
        'Neurocognitive': ['DELIRDEMFLG'],
        'Personality_Disorders': ['PERSONFLG'],
        'Other_Conditions': ['OTHERDISFLG']
    })

    def initialize_directories(self):
        """Create necessary output directories."""
        os.makedirs(self.OUTPUT_DIRECTORY, exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIRECTORY, "figures"), exist_ok=True)
        os.makedirs(os.path.join(self.OUTPUT_DIRECTORY, "tables"), exist_ok=True)
        return self


# Initialize configuration
CONFIG = AnalysisConfiguration().initialize_directories()
np.random.seed(CONFIG.RANDOM_SEED)

# ==============================================================================
# 5. FILE PATH MANAGEMENT
# ==============================================================================

def get_figure_path(filename: str, extension: str = DEFAULT_FILE_EXTENSION) -> str:
    """Generate full path for figure files."""
    return os.path.join(CONFIG.OUTPUT_DIRECTORY, "figures", f"{filename}.{extension}")


def get_table_path(filename: str) -> str:
    """Generate full path for data tables."""
    return os.path.join(CONFIG.OUTPUT_DIRECTORY, "tables", f"{filename}.csv")


def get_json_path(filename: str) -> str:
    """Generate full path for JSON metadata files."""
    return os.path.join(CONFIG.OUTPUT_DIRECTORY, f"{filename}.json")

# ==============================================================================
# 6. CORE UTILITY FUNCTIONS
# ==============================================================================

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce DataFrame memory usage through type optimization.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory-optimized DataFrame
    """
    optimized_df = df.copy()
    for column in optimized_df.columns:
        if pd.api.types.is_float_dtype(optimized_df[column]):
            optimized_df[column] = optimized_df[column].astype('float32')
        elif pd.api.types.is_integer_dtype(optimized_df[column]):
            optimized_df[column] = pd.to_numeric(optimized_df[column], downcast='integer')
    return optimized_df


def convert_to_integer(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Convert specified columns to integer type with error handling.
    
    Args:
        df: Input DataFrame
        columns: List of column names to convert
        
    Returns:
        DataFrame with converted columns
    """
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
    return df


def save_current_figure(base_path_without_extension: str):
    """
    Save current matplotlib figure in multiple formats.
    
    Args:
        base_path_without_extension: Path without file extension
    """
    os.makedirs(os.path.dirname(base_path_without_extension), exist_ok=True)
    plt.tight_layout()
    
    if SAVE_PNG_FORMAT:
        png_path = base_path_without_extension + ".png"
        plt.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Saved figure: {png_path}")
    
    if SAVE_PDF_FORMAT:
        pdf_path = base_path_without_extension + ".pdf"
        plt.savefig(pdf_path, dpi=FIGURE_DPI, bbox_inches="tight")
        logger.info(f"Saved figure: {pdf_path}")
    
    plt.close()


def save_figure(base_path_without_extension: str):
    """Wrapper function for figure saving."""
    save_current_figure(base_path_without_extension)


def save_dataframe(df: pd.DataFrame, file_path: str):
    """
    Save DataFrame to CSV with logging.
    
    Args:
        df: DataFrame to save
        file_path: Destination path
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info(f"Saved table: {file_path}")


def add_bar_annotations(axes, format_string="{:.2f}", font_size=9, 
                       rotation=0, vertical_offset=2):
    """
    Add value annotations to bar plots.
    
    Args:
        axes: Matplotlib axes object
        format_string: Format string for values
        font_size: Annotation font size
        rotation: Text rotation angle
        vertical_offset: Vertical offset from bars
    """
    for bar in axes.patches:
        height = bar.get_height()
        if np.isfinite(height) and height != 0:
            axes.annotate(
                format_string.format(height),
                (bar.get_x() + bar.get_width() / 2., height),
                ha="center", va="bottom", fontsize=font_size, rotation=rotation,
                xytext=(0, vertical_offset), textcoords="offset points"
            )


def add_heatmap_annotations(matrix: np.ndarray, axes, format_string="{:.2f}", 
                           font_size=8):
    """
    Add value annotations to heatmap cells.
    
    Args:
        matrix: Data matrix
        axes: Matplotlib axes object
        format_string: Format string for values
        font_size: Annotation font size
    """
    n_rows, n_cols = matrix.shape
    mean_value = np.nanmean(matrix) if np.isfinite(matrix).any() else 0.0
    
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                text_color = 'black' if value <= mean_value else 'white'
                axes.text(
                    col_idx, row_idx, format_string.format(value),
                    ha='center', va='center', color=text_color, fontsize=font_size
                )

# ==============================================================================
# 7. DATA PROCESSING MODULE
# ==============================================================================

def sequential_hot_deck_imputation(
    data_frame: pd.DataFrame,
    excluded_columns: Optional[List[str]] = None,
    missing_value_code: Any = -9,
    missing_threshold: float = 0.6,
    processing_chunk_size: int = 100_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Implement sequential hot-deck imputation for missing data.
    
    Args:
        data_frame: Input DataFrame with missing values
        excluded_columns: Columns to exclude from imputation
        missing_value_code: Code representing missing values
        missing_threshold: Maximum missing proportion for column retention
        processing_chunk_size: Size of processing chunks for memory efficiency
        random_seed: Random seed for reproducibility
        
    Returns:
        Imputed DataFrame
    """
    logger.info("Executing sequential hot-deck imputation (chunked processing)")
    imputed_df = data_frame.copy()
    excluded_columns = excluded_columns or []
    random_generator = np.random.default_rng(random_seed)
    
    # Convert missing value codes to NaN
    imputed_df = imputed_df.replace(missing_value_code, np.nan)
    
    # Identify and remove columns with excessive missingness
    missing_proportions = imputed_df.isna().mean()
    columns_to_drop = [
        col for col in missing_proportions.index 
        if missing_proportions[col] > missing_threshold and col not in excluded_columns
    ]
    
    if columns_to_drop:
        logger.warning(
            f"Removing columns exceeding missing threshold {missing_threshold}: {columns_to_drop}"
        )
        imputed_df = imputed_df.drop(columns=columns_to_drop)
    
    # Identify columns requiring imputation
    columns_to_impute = [
        col for col in imputed_df.columns 
        if col not in excluded_columns and imputed_df[col].isna().any()
    ]
    
    last_valid_values = {col: None for col in columns_to_impute}
    total_rows = len(imputed_df)
    
    # Process data in chunks for memory efficiency
    for chunk_start in range(0, total_rows, processing_chunk_size):
        chunk_end = min(chunk_start + processing_chunk_size, total_rows)
        data_chunk = imputed_df.iloc[chunk_start:chunk_end].copy()
        
        for column in columns_to_impute:
            column_values = data_chunk[column].to_numpy()
            missing_mask = pd.isna(column_values)
            
            if not missing_mask.any():
                if len(column_values) and not pd.isna(column_values[-1]):
                    last_valid_values[column] = column_values[-1]
                continue
            
            valid_values = column_values[~missing_mask]
            if valid_values.size == 0:
                continue
                
            current_value = last_valid_values[column]
            
            # Impute missing values sequentially
            for i in range(len(column_values)):
                if pd.isna(column_values[i]):
                    if current_value is not None:
                        column_values[i] = current_value
                    else:
                        column_values[i] = random_generator.choice(valid_values)
                else:
                    current_value = column_values[i]
            
            last_valid_values[column] = current_value
            data_chunk[column] = column_values
        
        # Update original DataFrame with imputed chunk
        imputed_df.iloc[chunk_start:chunk_end, 
                       imputed_df.columns.get_indexer(data_chunk.columns)] = data_chunk.values
    
    logger.info("Sequential hot-deck imputation completed successfully")
    return imputed_df

# ==============================================================================
# 8. DATA MAPPING & CATEGORICAL PROCESSING
# ==============================================================================

# Categorical value mappings for demographic variables
CATEGORICAL_MAPPINGS = {
    'GENDER': {1: 'Male', 2: 'Female'},
    'RACE': {
        1: 'American Indian/Alaska Native',
        2: 'Asian', 
        3: 'Black/African American',
        4: 'Native Hawaiian/Pacific Islander',
        5: 'White', 
        6: 'More than one race'
    },
    'ETHNIC': {
        1: 'Mexican',
        2: 'Puerto Rican', 
        3: 'Other Hispanic or Latino Origin',
        4: 'Not of Hispanic or Latino Origin'
    },
    'EDUC': {
        1: 'Special Education',
        2: 'Grades 0-8', 
        3: 'Grades 9-11',
        4: 'Grade 12/GED', 
        5: 'Grade 12 Above'
    },
    'MARSTAT': {
        1: 'Single',
        2: 'Married', 
        3: 'Separated',
        4: 'Divorced/Widowed'
    },
    'SAP': {1: 'Yes', 2: 'No'},
    'SMISED': {1: 'SMI', 2: 'SED', 3: 'Not SMI/SED'}
}


def convert_age_to_developmental_group(age_code: int) -> str:
    """
    Convert numeric age codes to developmental stage categories.
    
    Args:
        age_code: Numeric age code from dataset
        
    Returns:
        Developmental stage category
    """
    age_group_mapping = {
        1: '0-11', 2: '12-14', 3: '15-17', 4: '18-20', 5: '21-24',
        6: '25-29', 7: '30-34', 8: '35-39', 9: '40-44', 10: '45-49',
        11: '50-54', 12: '55-59', 13: '60-64', 14: '65+'
    }
    
    age_label = age_group_mapping.get(age_code, None)
    
    # Map to developmental stages
    if age_label in ['0-11', '12-14', '15-17']:
        return 'Child/Teen'
    elif age_label in ['18-20', '21-24', '25-29']:
        return 'Youth'
    elif age_label in ['30-34', '35-39', '40-44']:
        return 'Young Adult'
    elif age_label in ['45-49', '50-54', '55-59']:
        return 'Middle Aged'
    elif age_label in ['60-64', '65+']:
        return 'Senior'
    else:
        return 'Unknown'


def apply_categorical_mappings(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Apply categorical value mappings and create age groups.
    
    Args:
        data_frame: Input DataFrame
        
    Returns:
        DataFrame with mapped categorical variables
    """
    processed_df = data_frame.copy()
    
    # Apply categorical mappings
    for column_name, mapping_dict in CATEGORICAL_MAPPINGS.items():
        if column_name in processed_df.columns:
            processed_df[column_name] = processed_df[column_name].replace(mapping_dict)
    
    # Create age groups if age data available
    if 'AGE' in processed_df.columns:
        processed_df['AGE_GROUP'] = processed_df['AGE'].apply(convert_age_to_developmental_group)
        processed_df = processed_df.drop(columns=['AGE'])
    elif 'AGE_GROUP' not in processed_df.columns:
        processed_df['AGE_GROUP'] = 'Unknown'
    
    return processed_df


def apply_sampling_strategy(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Apply inclusion criteria and balanced sampling.
    
    Args:
        data_frame: Input DataFrame
        
    Returns:
        Sampled DataFrame
    """
    # Apply inclusion criteria
    filtered_df = data_frame[data_frame['NUMMHS'].isin(CONFIG.NUM_DIAGNOSES_KEEP)].copy()
    logger.info(
        f"Applied inclusion criteria (NUMMHS in {CONFIG.NUM_DIAGNOSES_KEEP}): "
        f"n={len(filtered_df):,}"
    )
    
    # Balanced sampling across groups
    sampled_data = (
        filtered_df.groupby('NUMMHS', group_keys=False)
        .apply(lambda x: x.sample(
            n=min(len(x), CONFIG.SAMPLE_SIZE_PER_GROUP), 
            random_state=CONFIG.RANDOM_SEED
        ))
        .reset_index(drop=True)
    )
    
    logger.info(
        f"Balanced sampling completed: {sampled_data['NUMMHS'].value_counts().to_dict()}"
    )
    return sampled_data


def compute_poststratification_weights(
    population_df: pd.DataFrame, 
    sample_df: pd.DataFrame
) -> pd.Series:
    """
    Compute post-stratification weights for population representativeness.
    
    Args:
        population_df: Full population DataFrame
        sample_df: Sampled DataFrame
        
    Returns:
        Series of post-stratification weights
    """
    if not CONFIG.USE_POSTSTRATIFICATION_WEIGHTS:
        return pd.Series(1.0, index=sample_df.index)
    
    population_proportions = population_df['NUMMHS'].value_counts(normalize=True)
    sample_proportions = sample_df['NUMMHS'].value_counts(normalize=True)
    
    weight_mapping = (population_proportions / sample_proportions).to_dict()
    weights = sample_df['NUMMHS'].map(weight_mapping).astype(float)
    
    return weights / weights.mean()

# ==============================================================================
# 9. LATENT CLASS ANALYSIS IMPLEMENTATION
# ==============================================================================

# Numerical stability constant
NUMERICAL_EPSILON = 1e-12


def safe_logsumexp(matrix: np.ndarray, axis=1) -> np.ndarray:
    """
    Numerically stable computation of log-sum-exp.
    
    Args:
        matrix: Input matrix
        axis: Axis along which to compute
        
    Returns:
        Log-sum-exp result
    """
    max_values = np.max(matrix, axis=axis, keepdims=True)
    return (max_values + np.log(np.sum(np.exp(matrix - max_values), axis=axis, keepdims=True))).squeeze()


def compute_bernoulli_log_likelihood(
    data_matrix: np.ndarray, 
    mixture_proportions: np.ndarray, 
    success_probabilities: np.ndarray
) -> np.ndarray:
    """
    Compute log-likelihood for Bernoulli mixture model.
    
    Args:
        data_matrix: Binary data matrix (n_observations × n_features)
        mixture_proportions: Class mixture proportions
        success_probabilities: Bernoulli success probabilities per class
        
    Returns:
        Log-likelihood matrix
    """
    data_matrix = np.asarray(data_matrix, dtype=np.float64)
    success_probabilities = np.clip(success_probabilities, NUMERICAL_EPSILON, 1 - NUMERICAL_EPSILON)
    
    log_success = np.log(success_probabilities)
    log_failure = np.log(1 - success_probabilities)
    
    # Compute component log-likelihoods
    component_log_likelihood = data_matrix @ log_success.T + (1 - data_matrix) @ log_failure.T
    
    # Add log mixture proportions
    return component_log_likelihood + np.log(np.clip(mixture_proportions, NUMERICAL_EPSILON, 1.0))


def fit_bernoulli_mixture_model(
    data_matrix: np.ndarray,
    n_components: int,
    n_initializations: int = 5,
    max_iterations: int = 500,
    convergence_tolerance: float = 1e-6,
    random_state: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Fit Bernoulli mixture model using Expectation-Maximization algorithm.
    
    Args:
        data_matrix: Binary data matrix
        n_components: Number of mixture components
        n_initializations: Number of random initializations
        max_iterations: Maximum EM iterations
        convergence_tolerance: Convergence threshold
        random_state: Random seed for reproducibility
        verbose: Print progress information
        
    Returns:
        Dictionary with model parameters and fit statistics
    """
    n_observations, n_features = data_matrix.shape
    best_log_likelihood = -np.inf
    best_model = None
    random_generator = np.random.default_rng(random_state)
    
    for initialization in range(n_initializations):
        # Initialize mixture proportions uniformly
        mixture_proportions = np.ones(n_components) / n_components
        
        try:
            # Use K-means for intelligent initialization
            kmeans = KMeans(
                n_clusters=n_components, 
                n_init=10, 
                random_state=(None if random_state is None else int(random_state + initialization))
            )
            cluster_labels = kmeans.fit_predict(data_matrix)
            
            # Initialize success probabilities from cluster means
            success_probabilities = np.vstack([
                np.clip(data_matrix[cluster_labels == k].mean(axis=0), 1e-6, 1 - 1e-6) 
                if (cluster_labels == k).any()
                else random_generator.uniform(0.05, 0.95, size=n_features) 
                for k in range(n_components)
            ])
        except Exception:
            # Fallback to random initialization
            success_probabilities = random_generator.uniform(0.05, 0.95, size=(n_components, n_features))
        
        previous_log_likelihood = -np.inf
        
        # EM algorithm iterations
        for iteration in range(max_iterations):
            # E-step: Compute responsibilities
            log_responsibilities = compute_bernoulli_log_likelihood(
                data_matrix, mixture_proportions, success_probabilities
            )
            log_normalization = safe_logsumexp(log_responsibilities, axis=1)
            current_log_likelihood = log_normalization.sum()
            
            responsibilities = np.exp(log_responsibilities - log_normalization[:, None])
            
            # M-step: Update parameters
            component_counts = responsibilities.sum(axis=0) + NUMERICAL_EPSILON
            mixture_proportions = component_counts / n_observations
            success_probabilities = (responsibilities.T @ data_matrix) / component_counts[:, None]
            success_probabilities = np.clip(success_probabilities, 1e-6, 1 - 1e-6)
            
            # Check convergence
            if abs(current_log_likelihood - previous_log_likelihood) < convergence_tolerance * (abs(previous_log_likelihood) + NUMERICAL_EPSILON):
                break
                
            previous_log_likelihood = current_log_likelihood
        
        # Compute assignment and model statistics
        assignments = responsibilities.argmax(axis=1)
        entropy = -np.sum(responsibilities * np.log(responsibilities + NUMERICAL_EPSILON), axis=1)
        normalized_entropy = 1.0 - entropy.mean() / np.log(n_components)
        mean_max_probability = responsibilities.max(axis=1).mean()
        
        # Information criteria
        n_parameters = n_components * n_features + (n_components - 1)
        aic = 2 * n_parameters - 2 * current_log_likelihood
        bic = np.log(n_observations) * n_parameters - 2 * current_log_likelihood
        
        # Update best model
        if current_log_likelihood > best_log_likelihood:
            best_log_likelihood = current_log_likelihood
            best_model = {
                'n_components': n_components,
                'mixture_proportions': mixture_proportions,
                'success_probabilities': success_probabilities,
                'posterior_probabilities': responsibilities,
                'responsibilities': responsibilities,
                'assignments': assignments,
                'log_likelihood': float(current_log_likelihood),
                'AIC': float(aic),
                'BIC': float(bic),
                'entropy': float(normalized_entropy),
                'mean_max_probability': float(mean_max_probability),
                'n_iterations': int(iteration + 1)
            }
    
    return best_model


def fit_latent_class_models(
    data_matrix: np.ndarray,
    component_range: range = CONFIG.CLASS_RANGE,
    n_initializations: int = CONFIG.NUMBER_INITIALIZATIONS,
    random_state: int = CONFIG.RANDOM_SEED,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    Fit multiple latent class models across specified component range.
    
    Args:
        data_matrix: Binary data matrix
        component_range: Range of components to evaluate
        n_initializations: Number of initializations per model
        random_state: Random seed for reproducibility
        verbose: Print progress information
        
    Returns:
        DataFrame with fit statistics and dictionary of fitted models
    """
    fit_statistics = []
    fitted_models = {}
    
    for n_components in component_range:
        model = fit_bernoulli_mixture_model(
            data_matrix, n_components, n_initializations=n_initializations,
            max_iterations=CONFIG.MAXIMUM_ITERATIONS,
            convergence_tolerance=CONFIG.CONVERGENCE_TOLERANCE,
            random_state=random_state
        )
        fitted_models[n_components] = model
        
        fit_statistics.append({
            'n_components': n_components,
            'log_likelihood': model['log_likelihood'],
            'AIC': model['AIC'],
            'BIC': model['BIC'],
            'entropy': model['entropy'],
            'mean_max_probability': model['mean_max_probability']
        })
    
    metrics_dataframe = pd.DataFrame(fit_statistics)
    return metrics_dataframe.sort_values('n_components').reset_index(drop=True), fitted_models

# ==============================================================================
# 10. NETWORK ANALYSIS MODULE
# ==============================================================================

def compute_cooccurrence_networks(diagnosis_flags: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute co-occurrence and Jaccard similarity matrices.
    
    Args:
        diagnosis_flags: DataFrame with binary diagnosis indicators
        
    Returns:
        Tuple of (co-occurrence counts, Jaccard similarity)
    """
    binary_matrix = diagnosis_flags.astype(int)
    
    # Compute co-occurrence counts
    cooccurrence_matrix = binary_matrix.T @ binary_matrix
    np.fill_diagonal(cooccurrence_matrix.values, 0)
    
    # Compute Jaccard similarity
    individual_sums = binary_matrix.sum(axis=0).to_numpy()
    union_matrix = (
        individual_sums[None, :] + individual_sums[:, None] - cooccurrence_matrix.to_numpy()
    )
    safe_denominator = np.where(union_matrix == 0, 1, union_matrix)
    jaccard_matrix = cooccurrence_matrix.to_numpy() / safe_denominator
    
    jaccard_dataframe = pd.DataFrame(
        jaccard_matrix, 
        index=binary_matrix.columns, 
        columns=binary_matrix.columns
    )
    
    return cooccurrence_matrix, jaccard_dataframe


def construct_comorbidity_network(
    cooccurrence_matrix: pd.DataFrame, 
    normalize: bool = True, 
    min_edge_support: int = CONFIG.MINIMUM_EDGE_SUPPORT
) -> Tuple[nx.Graph, Dict]:
    """
    Construct network graph from co-occurrence matrix.
    
    Args:
        cooccurrence_matrix: Matrix of disorder co-occurrences
        normalize: Whether to normalize edge weights
        min_edge_support: Minimum co-occurrence count for edge inclusion
        
    Returns:
        Tuple of (network graph, community partition)
    """
    filtered_matrix = cooccurrence_matrix.copy()
    original_counts = filtered_matrix.copy()
    
    # Apply minimum support threshold
    filtered_matrix.values[filtered_matrix.values < min_edge_support] = 0
    
    # Create network graph
    network_graph = nx.from_pandas_adjacency(filtered_matrix)
    
    # Add edge attributes
    for node_u, node_v in network_graph.edges():
        network_graph[node_u][node_v]["cooccurrence_count"] = int(original_counts.loc[node_u, node_v])
        
        max_value = original_counts.values.max() if original_counts.values.size else 1
        if normalize and max_value > 0:
            network_graph[node_u][node_v]["weight"] = float(original_counts.loc[node_u, node_v] / max_value)
        else:
            network_graph[node_u][node_v]["weight"] = float(original_counts.loc[node_u, node_v])
    
    # Remove isolated nodes
    isolated_nodes = [node for node, degree in dict(network_graph.degree()).items() if degree == 0]
    network_graph.remove_nodes_from(isolated_nodes)
    
    # Detect communities
    try:
        import community as community_louvain
        community_partition = community_louvain.best_partition(network_graph, weight='weight')
    except Exception as error:
        logger.warning(f"Louvain community detection failed: {error}; using single community")
        community_partition = {node: 0 for node in network_graph.nodes}
    
    nx.set_node_attributes(network_graph, community_partition, 'community')
    
    return network_graph, community_partition

# ==============================================================================
# 11. DSM-5 CLUSTER MAPPING
# ==============================================================================

def map_to_dsm5_clusters(diagnosis_flags: pd.DataFrame) -> pd.DataFrame:
    """
    Map individual diagnoses to DSM-5 diagnostic clusters.
    
    Args:
        diagnosis_flags: DataFrame with binary diagnosis indicators
        
    Returns:
        DataFrame with DSM-5 cluster indicators
    """
    cluster_indicators = pd.DataFrame(index=diagnosis_flags.index)
    
    for cluster_name, diagnosis_columns in CONFIG.DSM5_CLUSTER_DEFINITIONS.items():
        valid_columns = [col for col in diagnosis_columns if col in diagnosis_flags.columns]
        
        if valid_columns:
            cluster_indicators[f'DSM5_{cluster_name}'] = diagnosis_flags[valid_columns].max(axis=1)
            cluster_indicators[f'DSM5_{cluster_name}_count'] = diagnosis_flags[valid_columns].sum(axis=1)
    
    return cluster_indicators

# ==============================================================================
# 12. DEMOGRAPHIC ANALYSIS MODULE
# ==============================================================================

def compute_cramers_v(chi_square_statistic: float, sample_size: int, 
                     n_rows: int, n_columns: int) -> float:
    """
    Compute Cramér's V statistic for association strength.
    
    Args:
        chi_square_statistic: Chi-square test statistic
        sample_size: Total sample size
        n_rows: Number of rows in contingency table
        n_columns: Number of columns in contingency table
        
    Returns:
        Cramér's V value
    """
    if min(n_rows, n_columns) > 1:
        return np.sqrt(chi_square_statistic / (sample_size * (min(n_rows, n_columns) - 1)))
    else:
        return np.nan


def analyze_demographic_associations(
    data_frame: pd.DataFrame, 
    latent_class_column: str = 'Latent_Class'
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze associations between latent classes and demographic variables.
    
    Args:
        data_frame: DataFrame with latent class assignments and demographics
        latent_class_column: Name of latent class assignment column
        
    Returns:
        Dictionary with association statistics for each demographic variable
    """
    association_results = {}
    
    for demographic_column in [col for col in CONFIG.DEMOGRAPHIC_COLUMNS if col in data_frame.columns]:
        contingency_table = pd.crosstab(data_frame[latent_class_column], data_frame[demographic_column])
        total_sample_size = contingency_table.values.sum()
        
        try:
            chi2_statistic, p_value, degrees_of_freedom, expected_counts = chi2_contingency(contingency_table)
            
            # Check for small expected counts
            if (expected_counts < 5).sum() > 0:
                logger.warning(
                    f"{demographic_column}: {((expected_counts < 5).sum())} cells with expected count < 5; "
                    f"interpret chi-square with caution"
                )
        except Exception:
            chi2_statistic = p_value = degrees_of_freedom = np.nan
        
        cramers_v_value = compute_cramers_v(chi2_statistic, total_sample_size, *contingency_table.shape) \
            if not np.isnan(chi2_statistic) else np.nan
        
        association_results[demographic_column] = {
            'contingency_table': contingency_table,
            'chi_square': float(chi2_statistic) if not np.isnan(chi2_statistic) else None,
            'p_value': float(p_value) if not np.isnan(p_value) else None,
            'cramers_v': float(cramers_v_value) if not np.isnan(cramers_v_value) else None
        }
    
    # Apply False Discovery Rate correction
    p_values = [results['p_value'] for results in association_results.values() 
                if results['p_value'] is not None]
    
    if p_values:
        _, adjusted_p_values, _, _ = multipletests(p_values, method='fdr_bh')
        adjustment_index = 0
        
        for demographic_var, results in association_results.items():
            if results['p_value'] is not None:
                results['adjusted_p_value'] = float(adjusted_p_values[adjustment_index])
                adjustment_index += 1
            else:
                results['adjusted_p_value'] = None
    
    return association_results


def analyze_intersectional_associations(
    data_frame: pd.DataFrame,
    primary_factor: str = 'GENDER',
    secondary_factor: str = 'RACE',
    target_variable: str = 'Latent_Class'
) -> Dict[str, Any]:
    """
    Analyze intersectional associations between multiple demographic factors.
    
    Args:
        data_frame: Input DataFrame
        primary_factor: First demographic factor
        secondary_factor: Second demographic factor
        target_variable: Target variable for association analysis
        
    Returns:
        Dictionary with intersectional association statistics
    """
    if (primary_factor not in data_frame.columns) or (secondary_factor not in data_frame.columns):
        return {"analysis_successful": False, "reason": "Required columns not present"}
    
    intersection_table = pd.crosstab(
        index=[data_frame[primary_factor], data_frame[secondary_factor]], 
        columns=data_frame[target_variable]
    )
    total_sample_size = intersection_table.values.sum()
    
    chi2_statistic, p_value, degrees_of_freedom, expected_counts = chi2_contingency(intersection_table)
    
    return {
        "analysis_successful": True,
        "chi_square": float(chi2_statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": int(degrees_of_freedom),
        "sample_size": int(total_sample_size),
        "contingency_table": intersection_table
    }

# ==============================================================================
# 13. MODEL VALIDATION & STABILITY ANALYSIS
# ==============================================================================

def compute_classification_quality_metrics(
    model: dict, 
    confidence_threshold: float = CONFIG.CLASSIFICATION_CONFIDENCE_THRESHOLD
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Compute classification quality and boundary metrics.
    
    Args:
        model: Fitted LCA model
        confidence_threshold: Minimum posterior probability for confident classification
        
    Returns:
        Tuple of (quality metrics, maximum probabilities, entropy values)
    """
    posterior_probabilities = model['posterior_probabilities']
    maximum_probabilities = posterior_probabilities.max(axis=1)
    entropy_values = -np.sum(posterior_probabilities * np.log(posterior_probabilities + 1e-12), axis=1)
    
    metrics = {
        'proportion_uncertain_classifications': float((maximum_probabilities < confidence_threshold).mean()),
        'proportion_confident_classifications': float((maximum_probabilities >= confidence_threshold).mean()),
        'mean_classification_confidence': float(maximum_probabilities.mean()),
        'mean_classification_entropy': float(entropy_values.mean())
    }
    
    return metrics, maximum_probabilities, entropy_values


def perform_cross_validation(
    data_matrix: np.ndarray, 
    reference_model: dict, 
    n_folds: int = 5
) -> Dict[str, float]:
    """
    Perform k-fold cross-validation for model stability assessment.
    
    Args:
        data_matrix: Binary data matrix
        reference_model: Reference model for comparison
        n_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with cross-validation statistics
    """
    n_components = reference_model['n_components']
    cross_validator = StratifiedKFold(
        n_splits=n_folds, 
        shuffle=True, 
        random_state=CONFIG.RANDOM_SEED
    )
    
    adjusted_rand_indices = []
    test_log_likelihoods = []
    
    for train_indices, test_indices in cross_validator.split(data_matrix, reference_model['assignments']):
        train_data = data_matrix[train_indices]
        test_data = data_matrix[test_indices]
        
        # Fit model on training fold
        fold_model = fit_bernoulli_mixture_model(
            train_data, n_components, 
            n_initializations=CONFIG.NUMBER_INITIALIZATIONS,
            max_iterations=CONFIG.MAXIMUM_ITERATIONS,
            convergence_tolerance=CONFIG.CONVERGENCE_TOLERANCE,
            random_state=CONFIG.RANDOM_SEED
        )
        
        # Compute test log-likelihood
        test_log_responsibilities = compute_bernoulli_log_likelihood(
            test_data, fold_model['mixture_proportions'], fold_model['success_probabilities']
        )
        test_log_normalization = safe_logsumexp(test_log_responsibilities, axis=1)
        test_log_likelihoods.append(float(test_log_normalization.sum()))
        
        # Compute assignment similarity
        test_assignments = test_log_responsibilities.argmax(axis=1)
        adjusted_rand_indices.append(
            adjusted_rand_score(reference_model['assignments'][test_indices], test_assignments)
        )
    
    return {
        'cross_validation_ari_mean': float(np.mean(adjusted_rand_indices)),
        'cross_validation_ari_std': float(np.std(adjusted_rand_indices)),
        'cross_validation_log_likelihood_mean': float(np.mean(test_log_likelihoods)),
        'cross_validation_log_likelihood_std': float(np.std(test_log_likelihoods))
    }


def _bootstrap_iteration(
    data_matrix: np.ndarray, 
    n_components: int, 
    random_seed: int
) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    """
    Single bootstrap iteration for stability assessment.
    
    Args:
        data_matrix: Original data matrix
        n_components: Number of latent components
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (bootstrap indices, assignments, success status)
    """
    random_generator = np.random.default_rng(random_seed)
    bootstrap_indices = random_generator.integers(0, data_matrix.shape[0], size=data_matrix.shape[0])
    bootstrap_sample = data_matrix[bootstrap_indices]
    
    try:
        bootstrap_model = fit_bernoulli_mixture_model(
            bootstrap_sample, n_components, 
            n_initializations=max(2, CONFIG.NUMBER_INITIALIZATIONS // 2),
            max_iterations=CONFIG.MAXIMUM_ITERATIONS,
            convergence_tolerance=CONFIG.CONVERGENCE_TOLERANCE,
            random_state=random_seed
        )
        return bootstrap_indices, bootstrap_model['assignments'], True
    except Exception as error:
        logger.warning(f"Bootstrap iteration failed: {error}")
        return bootstrap_indices, None, False


def assess_bootstrap_stability(
    data_matrix: np.ndarray, 
    reference_model: dict, 
    n_bootstrap_samples: int = CONFIG.BOOTSTRAP_ITERATIONS
) -> Dict[str, float]:
    """
    Assess model stability through bootstrap resampling.
    
    Args:
        data_matrix: Binary data matrix
        reference_model: Reference model for comparison
        n_bootstrap_samples: Number of bootstrap samples
        
    Returns:
        Dictionary with bootstrap stability statistics
    """
    reference_assignments = reference_model['assignments']
    
    bootstrap_results = Parallel(n_jobs=CONFIG.PARALLEL_JOBS)(
        delayed(_bootstrap_iteration)(
            data_matrix, reference_model['n_components'], CONFIG.RANDOM_SEED + i
        ) for i in range(n_bootstrap_samples)
    )
    
    adjusted_rand_indices = []
    
    for bootstrap_indices, bootstrap_assignments, success in bootstrap_results:
        if success and bootstrap_assignments is not None:
            adjusted_rand_indices.append(
                adjusted_rand_score(reference_assignments[bootstrap_indices], bootstrap_assignments)
            )
    
    return {
        'bootstrap_ari_mean': float(np.mean(adjusted_rand_indices)) if adjusted_rand_indices else float('nan'),
        'bootstrap_ari_std': float(np.std(adjusted_rand_indices)) if adjusted_rand_indices else float('nan'),
        'successful_bootstrap_iterations': int(len(adjusted_rand_indices))
    }


def compute_lca_network_alignment(
    latent_class_assignments: Sequence[int], 
    network_module_assignments: Sequence[int]
) -> Dict[str, float]:
    """
    Compute alignment between LCA classes and network communities.
    
    Args:
        latent_class_assignments: Latent class assignments
        network_module_assignments: Network community assignments
        
    Returns:
        Dictionary with alignment metrics
    """
    return {
        'adjusted_rand_index': float(adjusted_rand_score(latent_class_assignments, network_module_assignments)),
        'normalized_mutual_information': float(
            normalized_mutual_info_score(latent_class_assignments, network_module_assignments)
        )
    }

# ==============================================================================
# 14. INTEGRATION ANALYSIS: LCA & NETWORK ALIGNMENT
# ==============================================================================

def compute_class_module_association_matrix(
    data_frame: pd.DataFrame, 
    latent_class_column: str, 
    network_partition: dict, 
    diagnosis_columns: list
) -> pd.DataFrame:
    """
    Compute association matrix between latent classes and network modules.
    
    Args:
        data_frame: DataFrame with latent class assignments
        latent_class_column: Name of latent class column
        network_partition: Network community assignments
        diagnosis_columns: List of diagnosis flag columns
        
    Returns:
        Association matrix (classes × modules)
    """
    disorder_modules = pd.Series(network_partition, name='module')\
        .reindex(diagnosis_columns)\
        .dropna()
    
    unique_classes = sorted(data_frame[latent_class_column].unique())
    unique_modules = sorted(disorder_modules.unique())
    
    association_matrix = pd.DataFrame(0.0, index=unique_classes, columns=unique_modules)
    
    for class_label in unique_classes:
        class_mask = data_frame[latent_class_column] == class_label
        
        for module_label in unique_modules:
            module_disorders = disorder_modules[disorder_modules == module_label].index.tolist()
            
            if module_disorders:
                association_matrix.loc[class_label, module_label] = (
                    data_frame.loc[class_mask, module_disorders].sum(axis=1) > 0
                ).mean()
    
    return association_matrix

# ==============================================================================
# 15. DIAGNOSTICS & SENSITIVITY ANALYSIS
# ==============================================================================

def assess_local_independence(
    data_frame: pd.DataFrame,
    diagnosis_columns: List[str],
    class_assignment_column: str = "Latent_Class"
) -> pd.DataFrame:
    """
    Assess local independence assumption within latent classes.
    
    Args:
        data_frame: DataFrame with class assignments and diagnoses
        diagnosis_columns: List of diagnosis indicator columns
        class_assignment_column: Name of class assignment column
        
    Returns:
        DataFrame with local independence diagnostics
    """
    diagnostic_results = []
    unique_classes = sorted(data_frame[class_assignment_column].unique())
    
    for class_label in unique_classes:
        class_subset = data_frame[data_frame[class_assignment_column] == class_label]
        
        if class_subset.empty:
            diagnostic_results.append((class_label, np.nan, 0))
            continue
        
        correlation_matrix = class_subset[diagnosis_columns].corr().values.astype(float)
        np.fill_diagonal(correlation_matrix, np.nan)
        mean_absolute_correlation = np.nanmean(np.abs(correlation_matrix))
        
        diagnostic_results.append((class_label, mean_absolute_correlation, len(class_subset)))
    
    return pd.DataFrame(
        diagnostic_results, 
        columns=["Class", "MeanAbsoluteResidualCorrelation", "SampleSize"]
    )


def assess_entropy_stability(
    reference_model: dict,
    original_data: np.ndarray,
    noise_proportion: float = 0.02
) -> Dict[str, Any]:
    """
    Assess classification stability under data perturbation.
    
    Args:
        reference_model: Reference LCA model
        original_data: Original binary data matrix
        noise_proportion: Proportion of data to perturb
        
    Returns:
        Dictionary with stability metrics
    """
    random_generator = np.random.default_rng(CONFIG.RANDOM_SEED)
    perturbed_data = original_data.copy()
    
    # Introduce random noise
    perturbation_mask = random_generator.uniform(size=perturbed_data.shape) < noise_proportion
    perturbed_data[perturbation_mask] = 1 - perturbed_data[perturbation_mask]
    
    # Compute assignments on perturbed data
    reference_assignments = reference_model['assignments']
    perturbed_log_responsibilities = compute_bernoulli_log_likelihood(
        perturbed_data, reference_model['mixture_proportions'], reference_model['success_probabilities']
    )
    perturbed_assignments = perturbed_log_responsibilities.argmax(axis=1)
    
    alignment_index = adjusted_rand_score(reference_assignments, perturbed_assignments)
    
    return {
        "noise_proportion": noise_proportion,
        "alignment_index": float(alignment_index),
        "sample_size": int(original_data.shape[0])
    }


def assess_community_detection_sensitivity(
    reference_network: nx.Graph,
    edge_thresholds: Tuple[int, ...] = (10, 25, 50)
) -> pd.DataFrame:
    """
    Assess sensitivity of community detection to edge threshold variations.
    
    Args:
        reference_network: Reference network graph
        edge_thresholds: Different edge weight thresholds to test
        
    Returns:
        DataFrame with sensitivity results
    """
    sensitivity_results = []
    
    # Check for Leiden availability
    leiden_available = False
    try:
        import igraph as ig
        import leidenalg  # noqa: F401
        leiden_available = True
    except Exception:
        logger.warning("Leiden algorithm not available; skipping Leiden comparison")
    
    def convert_partition_to_communities(partition_dictionary):
        """Convert partition dictionary to community sets."""
        community_structure = {}
        for node, community_id in partition_dictionary.items():
            community_structure.setdefault(community_id, set()).add(node)
        return list(community_structure.values())
    
    for threshold in edge_thresholds:
        # Filter edges by threshold
        if nx.get_edge_attributes(reference_network, "cooccurrence_count"):
            filtered_network = nx.Graph([
                (node_u, node_v, attributes) 
                for node_u, node_v, attributes in reference_network.edges(data=True)
                if attributes.get("cooccurrence_count", 0) >= threshold
            ])
        else:
            filtered_network = reference_network.copy()
        
        # Skip if network is too sparse
        if filtered_network.number_of_edges() == 0 or filtered_network.number_of_nodes() < 3:
            sensitivity_results.append({
                "Threshold": threshold, 
                "Algorithm": "Louvain", 
                "Modularity": np.nan,
                "Nodes": filtered_network.number_of_nodes(), 
                "Edges": filtered_network.number_of_edges()
            })
            continue
        
        # Louvain community detection
        try:
            import community as community_louvain
            louvain_partition = community_louvain.best_partition(filtered_network, weight="weight")
        except Exception:
            louvain_communities = nx.algorithms.community.greedy_modularity_communities(
                filtered_network, weight="weight"
            )
            louvain_partition = {
                node: community_id 
                for community_id, community in enumerate(louvain_communities) 
                for node in community
            }
        
        louvain_modularity = nx.algorithms.community.quality.modularity(
            filtered_network, 
            convert_partition_to_communities(louvain_partition), 
            weight="weight"
        )
        
        sensitivity_results.append({
            "Threshold": threshold, 
            "Algorithm": "Louvain", 
            "Modularity": float(louvain_modularity),
            "Nodes": filtered_network.number_of_nodes(), 
            "Edges": filtered_network.number_of_edges()
        })
        
        # Leiden community detection (if available)
        if leiden_available:
            import igraph as ig
            import leidenalg
            
            network_nodes = list(filtered_network.nodes())
            node_index_map = {node: index for index, node in enumerate(network_nodes)}
            
            igraph_edges = [
                (node_index_map[node_u], node_index_map[node_v]) 
                for node_u, node_v in filtered_network.edges()
            ]
            
            igraph_network = ig.Graph()
            igraph_network.add_vertices(len(network_nodes))
            igraph_network.add_edges(igraph_edges)
            
            edge_weights = [
                filtered_network[node_u][node_v].get("weight", 1.0) 
                for node_u, node_v in filtered_network.edges()
            ]
            
            leiden_partition = leidenalg.find_partition(
                igraph_network, 
                leidenalg.RBConfigurationVertexPartition, 
                weights=edge_weights
            )
            
            leiden_partition_dict = {}
            for community_id, community in enumerate(leiden_partition):
                for node_index in community:
                    leiden_partition_dict[network_nodes[node_index]] = community_id
            
            leiden_modularity = nx.algorithms.community.quality.modularity(
                filtered_network, 
                convert_partition_to_communities(leiden_partition_dict), 
                weight="weight"
            )
            
            sensitivity_results.append({
                "Threshold": threshold, 
                "Algorithm": "Leiden", 
                "Modularity": float(leiden_modularity),
                "Nodes": filtered_network.number_of_nodes(), 
                "Edges": filtered_network.number_of_edges()
            })
    
    return pd.DataFrame(sensitivity_results)


def assess_edge_stability(
    diagnosis_data: pd.DataFrame,
    diagnosis_columns: List[str],
    n_bootstrap_samples: int = 100
) -> pd.DataFrame:
    """
    Assess stability of network edges through bootstrap resampling.
    
    Args:
        diagnosis_data: DataFrame with diagnosis indicators
        diagnosis_columns: List of diagnosis column names
        n_bootstrap_samples: Number of bootstrap iterations
        
    Returns:
        DataFrame with edge stability frequencies
    """
    from itertools import combinations
    
    # Create edge index mapping
    edge_combinations = list(combinations(diagnosis_columns, 2))
    edge_index_map = {tuple(sorted(edge)): index for index, edge in enumerate(edge_combinations)}
    
    edge_presence_counts = np.zeros(len(edge_index_map), dtype=float)
    total_sample_size = len(diagnosis_data)
    random_generator = np.random.default_rng(CONFIG.RANDOM_SEED)
    
    for _ in range(n_bootstrap_samples):
        bootstrap_indices = random_generator.integers(0, total_sample_size, size=total_sample_size)
        bootstrap_sample = diagnosis_data.iloc[bootstrap_indices][diagnosis_columns].astype(int)
        
        # Compute co-occurrence in bootstrap sample
        bootstrap_cooccurrence = bootstrap_sample.T.dot(bootstrap_sample).values.astype(float)
        np.fill_diagonal(bootstrap_cooccurrence, 0.0)
        
        # Count edge presence
        edge_index = 0
        for i, diagnosis_a in enumerate(diagnosis_columns[:-1]):
            for j, diagnosis_b in enumerate(diagnosis_columns[i+1:], start=i+1):
                if bootstrap_cooccurrence[i, j] > 0:
                    edge_presence_counts[edge_index] += 1
                edge_index += 1
    
    edge_frequencies = edge_presence_counts / float(n_bootstrap_samples)
    
    result_dataframe = pd.DataFrame({
        "edge": [f"{diagnosis_u}—{diagnosis_v}" for diagnosis_u, diagnosis_v in edge_combinations],
        "bootstrap_frequency": edge_frequencies
    })
    
    return result_dataframe.sort_values("bootstrap_frequency", ascending=False).reset_index(drop=True)

# ==============================================================================
# 16. PUBLICATION-QUALITY VISUALIZATION FUNCTIONS
# ==============================================================================

def visualize_lca_model_selection(
    metrics_dataframe: pd.DataFrame, 
    optimal_components: int
):
    """
    Create comprehensive visualization of LCA model selection metrics.
    
    Args:
        metrics_dataframe: DataFrame with model fit statistics
        optimal_components: Optimal number of components selected
    """
    component_values = metrics_dataframe['n_components'].astype(int).to_numpy()
    bic_values = metrics_dataframe['BIC'].to_numpy()
    aic_values = metrics_dataframe['AIC'].to_numpy()
    entropy_values = metrics_dataframe['entropy'].to_numpy()
    max_prob_values = metrics_dataframe['mean_max_probability'].to_numpy()
    
    fig, primary_axis = plt.subplots(figsize=(8, 5))
    secondary_axis = primary_axis.twinx()
    
    color_scheme = get_color_palette(4, DIVERGING_PALETTE)
    
    # Plot information criteria
    primary_axis.plot(
        component_values, bic_values, marker='o', label='BIC', color=color_scheme[0]
    )
    primary_axis.plot(
        component_values, aic_values, marker='s', label='AIC', color=color_scheme[1]
    )
    
    # Plot classification quality metrics
    secondary_axis.plot(
        component_values, entropy_values, marker='^', linestyle='--', 
        label='Entropy', color=color_scheme[2]
    )
    secondary_axis.plot(
        component_values, max_prob_values, marker='d', linestyle='--', 
        label='Mean Max Probability', color=color_scheme[3]
    )
    
    # Highlight optimal component count
    primary_axis.axvline(
        optimal_components, color=darken("#bbbbbb", 0.7), 
        linestyle=':', linewidth=1
    )
    
    primary_axis.set_xlabel("Number of Latent Classes (K)")
    primary_axis.set_ylabel("Information Criteria (Lower Values Indicate Better Fit)")
    secondary_axis.set_ylabel("Classification Quality Metrics")
    primary_axis.set_title("Latent Class Analysis Model Selection")
    
    # Combine legends
    primary_lines = primary_axis.get_lines()
    secondary_lines = secondary_axis.get_lines()
    all_lines = primary_lines + secondary_lines
    
    valid_line_labels = [
        (line, line.get_label()) 
        for line in all_lines 
        if not line.get_label().startswith("_")
    ]
    
    primary_axis.legend(
        [line for line, _ in valid_line_labels],
        [label for _, label in valid_line_labels],
        frameon=False,
        loc="best"
    )
    
    plt.tight_layout()
    save_figure(os.path.splitext(get_figure_path("lca_model_selection", "png"))[0])


def create_publication_heatmap(
    data_frame: pd.DataFrame,
    plot_title: str,
    output_filename: str,
    value_format: str = "{:.2f}",
    color_map=None,
    custom_palette: Optional[List[str]] = None
):
    """
    Create publication-quality heatmap visualization.
    
    Args:
        data_frame: DataFrame to visualize
        plot_title: Title for the heatmap
        output_filename: Base filename for saving
        value_format: Format string for cell values
        color_map: Matplotlib colormap
        custom_palette: Custom color palette
    """
    from matplotlib.colors import ListedColormap
    
    fig, axes = plt.subplots(figsize=(8, 6))
    
    if custom_palette is not None:
        color_map = ListedColormap(custom_palette)
    
    heatmap = axes.imshow(data_frame.values, aspect='auto', cmap=color_map or "Blues")
    
    axes.set_xticks(np.arange(len(data_frame.columns)))
    axes.set_xticklabels(data_frame.columns, rotation=0, ha='right')
    axes.set_yticks(np.arange(len(data_frame.index)))
    axes.set_yticklabels(data_frame.index)
    axes.set_title(plot_title)
    
    fig.colorbar(heatmap, ax=axes, fraction=0.046, pad=0.04)
    add_heatmap_annotations(data_frame.values.astype(float), axes=axes, 
                           format_string=value_format, font_size=8)
    
    save_figure(os.path.splitext(get_figure_path(output_filename, "png"))[0])


def visualize_classification_confidence(maximum_probabilities: np.ndarray):
    """
    Visualize classification confidence distribution.
    
    Args:
        maximum_probabilities: Array of maximum posterior probabilities
    """
    color_scheme = get_color_palette(2, SEQUENTIAL_PALETTE)
    
    # Histogram of maximum probabilities
    fig, axes = plt.subplots(figsize=(7, 5))
    axes.hist(
        maximum_probabilities, bins=30, 
        color=color_scheme[0], edgecolor=darken(color_scheme[0])
    )
    axes.axvline(
        CONFIG.CLASSIFICATION_CONFIDENCE_THRESHOLD, 
        linestyle='--', color=darken("#bbbbbb", 0.7)
    )
    axes.set_title("Distribution of Maximum Posterior Probabilities")
    axes.set_xlabel("Maximum Posterior Probability")
    axes.set_ylabel("Frequency")
    save_figure(os.path.splitext(get_figure_path("classification_confidence_histogram", "png"))[0])
    
    # Cumulative confidence curve
    probability_thresholds = np.linspace(0, 1, 201)
    cumulative_confidence = [(maximum_probabilities >= threshold).mean() 
                            for threshold in probability_thresholds]
    
    fig, axes = plt.subplots(figsize=(7, 5))
    axes.plot(probability_thresholds, cumulative_confidence, 
             color=color_scheme[1], linewidth=2)
    axes.axvline(
        CONFIG.CLASSIFICATION_CONFIDENCE_THRESHOLD, 
        linestyle='--', color=darken("#bbbbbb", 0.7)
    )
    axes.set_title("Cumulative Classification Confidence")
    axes.set_xlabel("Confidence Threshold")
    axes.set_ylabel("Proportion of Samples Above Threshold")
    save_figure(os.path.splitext(get_figure_path("cumulative_confidence_curve", "png"))[0])


def visualize_demographic_association_strength(demographic_results: dict):
    """
    Visualize strength of demographic associations with latent classes.
    
    Args:
        demographic_results: Dictionary with demographic association statistics
    """
    if not demographic_results:
        return
    
    # Prepare data for visualization
    visualization_data = [
        (variable, results.get('cramers_v', np.nan), results.get('adjusted_p_value', np.nan))
        for variable, results in demographic_results.items()
    ]
    
    association_dataframe = pd.DataFrame(
        visualization_data, 
        columns=['demographic_variable', 'cramers_v', 'adjusted_p_value']
    ).sort_values('cramers_v', ascending=False)
    
    color_scheme = get_color_palette(len(association_dataframe), QUALITATIVE_PALETTE)
    
    # Cramér's V visualization
    fig, axes = plt.subplots(figsize=(8, 5))
    axes.bar(
        association_dataframe['demographic_variable'], 
        association_dataframe['cramers_v'],
        color=color_scheme, 
        edgecolor=[darken(color) for color in color_scheme]
    )
    axes.set_title("Association Strength with Latent Classes (Cramér's V)")
    axes.set_ylabel("Cramér's V Coefficient")
    axes.set_xticklabels(association_dataframe['demographic_variable'], rotation=0, ha='right')
    add_bar_annotations(axes, format_string="{:.2f}", font_size=9)
    save_figure(os.path.splitext(get_figure_path("demographic_association_strength", "png"))[0])
    
    # Statistical significance visualization
    fig, axes = plt.subplots(figsize=(8, 5))
    negative_log_pvalues = -np.log10(np.clip(
        association_dataframe['adjusted_p_value'].astype(float), 1e-300, 1.0
    ))
    
    significance_colors = get_color_palette(len(association_dataframe), QUALITATIVE_PALETTE)
    axes.bar(
        association_dataframe['demographic_variable'], 
        negative_log_pvalues,
        color=significance_colors, 
        edgecolor=[darken(color) for color in significance_colors]
    )
    axes.set_title("Statistical Significance of Demographic Associations")
    axes.set_ylabel("−log₁₀(Adjusted p-value)")
    axes.set_xticklabels(association_dataframe['demographic_variable'], rotation=0, ha='right')
    add_bar_annotations(axes, format_string="{:.2f}", font_size=9)
    save_figure(os.path.splitext(get_figure_path("demographic_significance", "png"))[0])


def visualize_comorbidity_network(network_graph: nx.Graph, community_partition: dict):
    """
    Create publication-quality network visualization.
    
    Args:
        network_graph: NetworkX graph object
        community_partition: Dictionary of node community assignments
    """
    if network_graph.number_of_nodes() == 0:
        return
    
    fig, axes = plt.subplots(figsize=(8, 6))
    
    # Network layout
    node_positions = nx.spring_layout(network_graph, seed=CONFIG.RANDOM_SEED)
    
    # Color nodes by community
    community_ids = sorted(pd.Series(community_partition).unique())
    community_colors = get_color_palette(len(community_ids), QUALITATIVE_PALETTE)
    
    node_color_map = [
        community_colors[community_ids.index(community_partition.get(node, community_ids[0]))]
        for node in network_graph.nodes()
    ]
    
    # Draw network components
    nx.draw_networkx_nodes(
        network_graph, node_positions, 
        node_color=node_color_map, node_size=520,
        ax=axes, edgecolors=darken("#cccccc")
    )
    
    # Draw edges with weight-based styling
    edge_weights = np.array([
        attributes.get('weight', 1.0) 
        for _, _, attributes in network_graph.edges(data=True)
    ])
    edge_widths = 0.5 + 2.5 * (
        edge_weights / (edge_weights.max() if len(edge_weights) and edge_weights.max() > 0 else 1)
    )
    
    nx.draw_networkx_edges(
        network_graph, node_positions, 
        width=edge_widths, alpha=0.6, ax=axes,
        edge_color=darken("#c7d3e5", 0.8)
    )
    
    # Node labels
    nx.draw_networkx_labels(network_graph, node_positions, font_size=8, ax=axes)
    
    # Community legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=f"Community {community_id}",
                    markerfacecolor=community_colors[i], markersize=10)
        for i, community_id in enumerate(community_ids)
    ]
    
    axes.legend(
        handles=legend_elements, 
        title="Louvain Communities", 
        loc="center left",
        bbox_to_anchor=(1.02, 0.5), 
        frameon=False
    )
    
    axes.set_title("Mental Health Comorbidity Network")
    axes.axis('off')
    plt.tight_layout()
    save_figure(os.path.splitext(get_figure_path("comorbidity_network", "png"))[0])


def visualize_conditional_probabilities(
    diagnosis_data: pd.DataFrame, 
    class_assignments: np.ndarray
):
    """
    Visualize conditional probabilities of disorders within latent classes.
    
    Args:
        diagnosis_data: DataFrame with diagnosis indicators
        class_assignments: Array of latent class assignments
    """
    analysis_data = diagnosis_data.copy()
    analysis_data['Latent_Class'] = class_assignments
    
    conditional_probabilities = analysis_data.groupby('Latent_Class')[diagnosis_data.columns].mean().T
    conditional_probabilities = conditional_probabilities.sort_index()
    
    save_dataframe(
        conditional_probabilities.reset_index().rename(columns={'index': 'Diagnosis'}), 
        get_table_path("conditional_probabilities_by_class")
    )
    
    plot_title = "Conditional Disorder Probabilities by Latent Class\nP(Disorder = 1 | Class)"
    create_publication_heatmap(
        conditional_probabilities, plot_title, "conditional_probabilities_heatmap",
        value_format="{:.2f}", custom_palette=SEQUENTIAL_PALETTE
    )


def visualize_dsm5_cluster_prevalence(
    dsm5_data: pd.DataFrame, 
    class_assignments: np.ndarray
):
    """
    Visualize DSM-5 cluster prevalence across latent classes.
    
    Args:
        dsm5_data: DataFrame with DSM-5 cluster indicators
        class_assignments: Array of latent class assignments
    """
    binary_cluster_columns = [
        column for column in dsm5_data.columns 
        if column.startswith("DSM5_") and not column.endswith("_count")
    ]
    
    if not binary_cluster_columns:
        return
    
    analysis_data = dsm5_data.copy()
    analysis_data['Latent_Class'] = class_assignments
    
    cluster_prevalence = analysis_data.groupby('Latent_Class')[binary_cluster_columns].mean().T
    cluster_prevalence = cluster_prevalence.sort_index()
    
    save_dataframe(
        cluster_prevalence.reset_index().rename(columns={'index': 'DSM5_Cluster'}), 
        get_table_path("dsm5_cluster_prevalence")
    )
    
    plot_title = "DSM-5 Diagnostic Cluster Prevalence by Latent Class"
    create_publication_heatmap(
        cluster_prevalence, plot_title, "dsm5_cluster_prevalence",
        value_format="{:.2f}", custom_palette=SEQUENTIAL_PALETTE
    )


def visualize_class_module_alignment(alignment_matrix: pd.DataFrame):
    """
    Visualize alignment between latent classes and network modules.
    
    Args:
        alignment_matrix: DataFrame with class-module association strengths
    """
    plot_title = "Latent Class – Network Module Alignment"
    create_publication_heatmap(
        alignment_matrix, plot_title, "class_module_alignment",
        value_format="{:.2f}", custom_palette=DIVERGING_PALETTE
    )


def visualize_lca_network_overlap(
    latent_class_labels: Sequence[int],
    network_module_labels: Sequence[int],
    class_names: Optional[Sequence[str]] = None,
    module_names: Optional[Sequence[str]] = None
):
    """
    Visualize overlap between LCA classes and network modules.
    
    Args:
        latent_class_labels: Latent class assignments
        network_module_labels: Network module assignments
        class_names: Optional class names for labeling
        module_names: Optional module names for labeling
    """
    latent_class_labels = np.asarray(latent_class_labels)
    network_module_labels = np.asarray(network_module_labels)
    
    overlap_table = pd.crosstab(latent_class_labels, network_module_labels, normalize="index")
    
    fig, axes = plt.subplots(figsize=(6, 4))
    heatmap = axes.imshow(overlap_table.values, aspect="auto", cmap="RdBu_r")
    
    axes.set_xlabel("Network Module")
    axes.set_ylabel("Latent Class")
    
    x_labels = [str(i) for i in overlap_table.columns] if module_names is None else list(module_names)
    y_labels = [str(i) for i in overlap_table.index] if class_names is None else list(class_names)
    
    axes.set_xticks(range(len(x_labels)))
    axes.set_xticklabels(x_labels)
    axes.set_yticks(range(len(y_labels)))
    axes.set_yticklabels(y_labels)
    
    # Add value annotations
    for row in range(overlap_table.shape[0]):
        for col in range(overlap_table.shape[1]):
            axes.text(
                col, row, f"{overlap_table.values[row, col]:.2f}", 
                ha="center", va="center", fontsize=8
            )
    
    plt.title("LCA–Network Alignment (Row-normalized Proportions)")
    save_figure(os.path.splitext(get_figure_path("lca_network_overlap", "png"))[0])


def create_analysis_pipeline_schematic():
    """Create visual schematic of the analysis pipeline."""
    plt.figure(figsize=(8, 4))
    schematic_axes = plt.gca()
    schematic_axes.axis("off")
    
    # Define pipeline components
    pipeline_components = [
        (0.05, 0.6, 0.22, 0.25, "Data Preparation\n& Imputation"),
        (0.35, 0.6, 0.22, 0.25, "Latent Class Analysis\n(Bernoulli Mixture Model)"),
        (0.35, 0.1, 0.22, 0.25, "Comorbidity Network\n(Jaccard + Community Detection)"),
        (0.65, 0.35, 0.28, 0.3, "Integration & Validation\nClass-Module Alignment\nSensitivity Analysis"),
    ]
    
    # Draw component boxes
    for (x_pos, y_pos, width, height, label) in pipeline_components:
        component_box = plt.Rectangle(
            (x_pos, y_pos), width, height, 
            fill=False, linewidth=2, edgecolor=darken("#cbd5e1", 0.8)
        )
        schematic_axes.add_patch(component_box)
        schematic_axes.text(x_pos + width / 2, y_pos + height / 2, label, 
                           ha="center", va="center")
    
    # Draw flow arrows
    schematic_axes.annotate("", xy=(0.35, 0.725), xytext=(0.27, 0.725), 
                           arrowprops=dict(arrowstyle="->"))
    schematic_axes.annotate("", xy=(0.46, 0.35), xytext=(0.46, 0.6), 
                           arrowprops=dict(arrowstyle="->"))
    schematic_axes.annotate("", xy=(0.57, 0.35), xytext=(0.57, 0.6), 
                           arrowprops=dict(arrowstyle="->"))
    schematic_axes.annotate("", xy=(0.65, 0.5), xytext=(0.57, 0.5), 
                           arrowprops=dict(arrowstyle="->"))
    
    plt.title("Comorbidity Analysis Pipeline Overview", pad=8)
    save_figure(os.path.splitext(get_figure_path("analysis_pipeline", "png"))[0])

# ==============================================================================
# 17. DEMOGRAPHIC COMPOSITE VISUALIZATION
# ==============================================================================

def compute_demographic_proportions(
    data_frame: pd.DataFrame, 
    demographic_column: str, 
    class_column: str = "Latent_Class"
) -> pd.DataFrame:
    """
    Compute demographic proportions within each latent class.
    
    Args:
        data_frame: DataFrame with demographic and class data
        demographic_column: Demographic variable to analyze
        class_column: Latent class assignment column
        
    Returns:
        DataFrame with proportions by class and demographic category
    """
    if demographic_column not in data_frame.columns:
        return pd.DataFrame()
    
    proportion_table = pd.crosstab(
        data_frame[class_column], data_frame[demographic_column], 
        normalize="index"
    )
    return proportion_table.sort_index()


def create_demographic_composite_visualization(
    data_frame: pd.DataFrame,
    demographic_columns: List[str],
    class_column: str = "Latent_Class",
    max_columns_per_row: int = 2,
    figure_scaling: float = 3.8,
    display_percentages: bool = True,
):
    """
    Create comprehensive visualization of demographic distributions across latent classes.
    
    Args:
        data_frame: DataFrame with demographic and class data
        demographic_columns: List of demographic variables to visualize
        class_column: Latent class assignment column
        max_columns_per_row: Maximum subplots per row
        figure_scaling: Base size scaling factor
        display_percentages: Whether to display percentages instead of proportions
    """
    valid_demographic_columns = [
        column for column in demographic_columns 
        if column in data_frame.columns
    ]
    
    if not valid_demographic_columns:
        logger.warning("No valid demographic columns found for composite visualization")
        return
    
    # Calculate subplot layout
    n_demographics = len(valid_demographic_columns)
    n_columns = min(max_columns_per_row, n_demographics)
    n_rows = math.ceil(n_demographics / n_columns)
    
    figure_width = max(10.0, n_columns * figure_scaling)
    figure_height = max(4.5, n_rows * (figure_scaling - 0.4))
    
    fig, subplot_axes = plt.subplots(
        n_rows, n_columns, 
        figsize=(figure_width, figure_height), 
        squeeze=False
    )
    
    # Demographic-specific color schemes
    DEMOGRAPHIC_COLOR_SCHEMES = {
        "AGE_GROUP": ["#d4d4d4", "#a4c4a8", "#79b9ba", "#f4b77e", "#e37a6f"],
        "GENDER": ["#c8c8c8", "#b292c0"],
        "RACE": ["#a0c2dd", "#a8b8b2", "#e0cdbd", "#8aa49a", "#d28c88", "#cfcfcf"],
        "SAP": ["#8ec1da", "#ededed"],
        "SMISED": ["#a7d3c2", "#68a8a1", "#37787c"],
        "EDUC": ["#f0d5a2", "#f5b971", "#b48d5a", "#df7b70", "#a94c48"],
        "MARSTAT": ["#e1c4e6", "#d592b8", "#ba709d", "#9e4f7e"],
        "ETHNIC": ["#d5c9df", "#b598c9", "#9474b1", "#714b8c"],
    }
    
    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.45, wspace=0.3)
    
    for demographic_index, demographic_variable in enumerate(valid_demographic_columns):
        row_index, col_index = divmod(demographic_index, n_columns)
        current_axes = subplot_axes[row_index][col_index]
        
        demographic_proportions = compute_demographic_proportions(
            data_frame, demographic_variable, class_column
        )
        
        if demographic_proportions.empty:
            current_axes.set_visible(False)
            continue
        
        # Select appropriate color scheme
        color_palette = DEMOGRAPHIC_COLOR_SCHEMES.get(
            demographic_variable, 
            get_color_palette(len(demographic_proportions.columns), QUALITATIVE_PALETTE)
        )
        category_colors = color_palette[:len(demographic_proportions.columns)]
        
        # Create stacked bar chart
        cumulative_heights = np.zeros(len(demographic_proportions))
        class_labels = [str(class_id) for class_id in demographic_proportions.index]
        
        for category_index, category_name in enumerate(demographic_proportions.columns):
            category_heights = demographic_proportions[category_name].to_numpy()
            if display_percentages:
                category_heights = category_heights * 100
            
            current_axes.bar(
                class_labels,
                category_heights,
                bottom=cumulative_heights,
                label=str(category_name),
                color=category_colors[category_index],
                edgecolor=darken(category_colors[category_index]),
                linewidth=0.5,
            )
            cumulative_heights += category_heights
        
        # Axes formatting
        current_axes.set_title(f"{demographic_variable} Distribution", fontsize=11)
        
        if row_index == n_rows - 1:
            current_axes.set_xlabel("Latent Class", fontsize=10)
        else:
            current_axes.set_xlabel("")
        
        y_axis_label = "Percentage (%)" if display_percentages else "Proportion"
        current_axes.set_ylabel(y_axis_label, fontsize=10)
        
        y_axis_limit = 105 if display_percentages else 1.05
        current_axes.set_ylim(0, y_axis_limit)
        
        # Grid and styling
        current_axes.grid(
            axis="y", linestyle=":", linewidth=0.5, 
            color="#dddddd", alpha=0.6
        )
        
        # Legend configuration
        n_categories = len(demographic_proportions.columns)
        legend_columns = min(n_categories, 3)
        legend_fontsize = 7 if n_categories > 5 else 8
        
        current_axes.legend(
            fontsize=legend_fontsize,
            title="Category",
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=legend_columns,
        )
        
        # Clean axis borders
        for border in ["top", "right"]:
            current_axes.spines[border].set_visible(False)
    
    # Hide unused subplots
    for subplot_index in range(n_rows * n_columns):
        if subplot_index >= n_demographics:
            row, col = divmod(subplot_index, n_columns)
            subplot_axes[row][col].set_visible(False)
    
    plt.suptitle(
        "Demographic Characteristics Across Latent Classes",
        y=1.03,
        fontsize=11.5,
    )
    plt.tight_layout()
    save_figure(
        os.path.splitext(get_figure_path("demographic_composite", "png"))[0]
    )

# ==============================================================================
# 18. MAIN ANALYSIS PIPELINE
# ==============================================================================

def execute_analysis_pipeline():
    """
    Execute the complete comorbidity analysis pipeline.
    
    This function coordinates all analysis steps from data loading through
    model validation and result export.
    """
    logger.info("Initiating Mental Health Comorbidity Analysis Pipeline")
    
    # ==========================================================================
    # 1. DATA LOADING & INITIAL PROCESSING
    # ==========================================================================
    
    if not os.path.exists(CONFIG.DATA_FILE_PATH):
        raise FileNotFoundError(f"Data file not found: {CONFIG.DATA_FILE_PATH}")
    
    raw_data = pd.read_csv(CONFIG.DATA_FILE_PATH)
    processed_data = optimize_dataframe_memory(raw_data)
    
    # ==========================================================================
    # 2. SAMPLE SELECTION & WEIGHTING
    # ==========================================================================
    
    analysis_sample = processed_data[
        processed_data['NUMMHS'].isin(CONFIG.NUM_DIAGNOSES_KEEP)
    ].copy()
    final_sample = apply_sampling_strategy(analysis_sample)
    
    poststratification_weights = compute_poststratification_weights(
        analysis_sample, final_sample
    )
    _ = poststratification_weights  # Reserved for future weighted analyses
    
    # ==========================================================================
    # 3. DATA IMPUTATION & QUALITY CONTROL
    # ==========================================================================
    
    excluded_columns = ['CASEID', 'MH1', 'MH2', 'MH3']
    imputed_data = sequential_hot_deck_imputation(
        final_sample,
        excluded_columns=[col for col in excluded_columns if col in final_sample.columns],
        missing_value_code=CONFIG.MISSING_DATA_CODE,
        missing_threshold=CONFIG.MISSING_DATA_THRESHOLD,
        processing_chunk_size=CONFIG.IMPUTATION_CHUNK_SIZE,
        random_seed=CONFIG.RANDOM_SEED
    )
    
    # ==========================================================================
    # 4. CATEGORICAL VARIABLE PROCESSING
    # ==========================================================================
    
    categorized_data = apply_categorical_mappings(imputed_data)
    
    # ==========================================================================
    # 5. DIAGNOSIS FLAG PROCESSING
    # ==========================================================================
    
    for diagnosis_column in CONFIG.DIAGNOSIS_FLAG_COLUMNS:
        if diagnosis_column in categorized_data.columns:
            categorized_data[diagnosis_column] = pd.to_numeric(
                categorized_data[diagnosis_column], errors='coerce'
            ).fillna(0).astype(int)
    
    diagnosis_flags = categorized_data[
        [col for col in CONFIG.DIAGNOSIS_FLAG_COLUMNS if col in categorized_data.columns]
    ].copy()
    
    # ==========================================================================
    # 6. NETWORK ANALYSIS: CO-OCCURRENCE & COMMUNITY DETECTION
    # ==========================================================================
    
    cooccurrence_matrix, jaccard_similarity = compute_cooccurrence_networks(diagnosis_flags)
    
    save_dataframe(
        cooccurrence_matrix.reset_index().rename(columns={'index': 'Diagnosis'}), 
        get_table_path("cooccurrence_matrix")
    )
    save_dataframe(
        jaccard_similarity.reset_index().rename(columns={'index': 'Diagnosis'}), 
        get_table_path("jaccard_similarity")
    )
    
    create_publication_heatmap(
        jaccard_similarity, 
        "Jaccard Similarity Matrix (Disorder Co-occurrence)", 
        "jaccard_similarity_heatmap", 
        value_format="{:.2f}"
    )
    
    log_cooccurrence = np.log1p(cooccurrence_matrix.replace(0, 0))
    create_publication_heatmap(
        log_cooccurrence, 
        "Co-occurrence Frequencies (Log-transformed)", 
        "cooccurrence_heatmap", 
        value_format="{:.2f}"
    )
    
    comorbidity_network, community_structure = construct_comorbidity_network(
        cooccurrence_matrix, 
        normalize=CONFIG.NORMALIZE_NETWORK_WEIGHTS, 
        min_edge_support=CONFIG.MINIMUM_EDGE_SUPPORT
    )
    
    pd.Series(community_structure).to_csv(
        get_table_path("network_communities"), 
        index=True, 
        header=["community"]
    )
    
    visualize_comorbidity_network(comorbidity_network, community_structure)
    
    # ==========================================================================
    # 7. LATENT CLASS ANALYSIS: MODEL SELECTION & FITTING
    # ==========================================================================
    
    diagnosis_matrix = diagnosis_flags.to_numpy(dtype=float)
    model_metrics, fitted_models = fit_latent_class_models(
        diagnosis_matrix, 
        component_range=CONFIG.CLASS_RANGE, 
        n_initializations=CONFIG.NUMBER_INITIALIZATIONS, 
        random_state=CONFIG.RANDOM_SEED
    )
    
    save_dataframe(model_metrics, get_table_path("lca_model_metrics"))
    
    optimal_components = int(model_metrics.loc[model_metrics['BIC'].idxmin(), 'n_components'])
    optimal_model = fitted_models[optimal_components]
    categorized_data['Latent_Class'] = optimal_model['assignments']
    
    visualize_lca_model_selection(model_metrics, optimal_components)
    
    # ==========================================================================
    # 8. CLASSIFICATION QUALITY ASSESSMENT
    # ==========================================================================
    
    classification_metrics, max_probabilities, entropy_values = compute_classification_quality_metrics(
        optimal_model, confidence_threshold=CONFIG.CLASSIFICATION_CONFIDENCE_THRESHOLD
    )
    
    with open(get_json_path("classification_quality"), 'w') as metrics_file:
        json.dump(classification_metrics, metrics_file, indent=2)
    
    visualize_classification_confidence(max_probabilities)
    
    # ==========================================================================
    # 9. DSM-5 CLUSTER MAPPING & ANALYSIS
    # ==========================================================================
    
    dsm5_indicators = map_to_dsm5_clusters(diagnosis_flags)
    save_dataframe(dsm5_indicators.reset_index(drop=True), get_table_path("dsm5_indicators"))
    
    visualize_conditional_probabilities(diagnosis_flags, optimal_model['assignments'])
    visualize_dsm5_cluster_prevalence(dsm5_indicators, optimal_model['assignments'])
    
    # ==========================================================================
    # 10. MODEL VALIDATION: CROSS-VALIDATION & BOOTSTRAP
    # ==========================================================================
    
    cross_validation_results = perform_cross_validation(diagnosis_matrix, optimal_model, n_folds=5)
    bootstrap_results = assess_bootstrap_stability(diagnosis_matrix, optimal_model, 
                                                 n_bootstrap_samples=CONFIG.BOOTSTRAP_ITERATIONS)
    
    # ==========================================================================
    # 11. DEMOGRAPHIC ASSOCIATION ANALYSIS
    # ==========================================================================
    
    demographic_associations = analyze_demographic_associations(
        categorized_data, latent_class_column='Latent_Class'
    )
    
    demographic_results = []
    for variable_name, association_stats in demographic_associations.items():
        demographic_results.append({
            "variable": variable_name,
            "chi_square": association_stats.get("chi_square"),
            "p_value": association_stats.get("p_value"),
            "adjusted_p_value": association_stats.get("adjusted_p_value"),
            "cramers_v": association_stats.get("cramers_v")
        })
        association_stats["contingency_table"].to_csv(get_table_path(f"contingency_{variable_name}"))
    
    save_dataframe(pd.DataFrame(demographic_results), get_table_path("demographic_associations"))
    visualize_demographic_association_strength(demographic_associations)
    
    # Composite demographic visualization
    create_demographic_composite_visualization(
        categorized_data, CONFIG.DEMOGRAPHIC_COLUMNS, 
        class_column="Latent_Class", max_columns_per_row = 2, figure_scaling = 3.8, display_percentages = True
    )
    
    # ==========================================================================
    # 12. INTERSECTIONAL DEMOGRAPHIC ANALYSIS
    # ==========================================================================
    
    intersection_analysis = analyze_intersectional_associations(
        categorized_data, 
        primary_factor="GENDER", 
        secondary_factor="RACE", 
        target_variable="Latent_Class"
    )
    
    if intersection_analysis.get("analysis_successful", False):
        intersection_analysis["contingency_table"].to_csv(get_table_path("intersection_gender_race"))
        with open(get_json_path("intersectional_stats"), "w") as stats_file:
            json.dump({
                key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
                for key, value in intersection_analysis.items() if key != "contingency_table"
            }, stats_file, indent=2)
    
    # ==========================================================================
    # 13. LCA-NETWORK INTEGRATION ANALYSIS
    # ==========================================================================
    
    diagnosis_module_assignments = pd.Series(community_structure, name='module')\
        .reindex([col for col in CONFIG.DIAGNOSIS_FLAG_COLUMNS if col in diagnosis_flags.columns])\
        .dropna()
    
    if not diagnosis_module_assignments.empty:
        module_dummy_variables = pd.get_dummies(diagnosis_module_assignments)
        patient_module_assignments = (diagnosis_flags[diagnosis_module_assignments.index] @ module_dummy_variables).idxmax(axis=1)
        
        alignment_statistics = compute_lca_network_alignment(
            categorized_data['Latent_Class'], patient_module_assignments
        )
        
        class_module_matrix = compute_class_module_association_matrix(
            categorized_data, 'Latent_Class', community_structure, 
            [col for col in CONFIG.DIAGNOSIS_FLAG_COLUMNS if col in diagnosis_flags.columns]
        )
        
        save_dataframe(
            class_module_matrix.reset_index().rename(columns={'index': 'Latent_Class'}), 
            get_table_path("class_module_association")
        )
        
        visualize_class_module_alignment(class_module_matrix)
        visualize_lca_network_overlap(categorized_data['Latent_Class'], patient_module_assignments)
    else:
        alignment_statistics = {"adjusted_rand_index": None, "normalized_mutual_information": None}
    
    # ==========================================================================
    # 14. REPRESENTATIVENESS ASSESSMENT
    # ==========================================================================
    
    representativeness_summary = {}
    for demographic_variable in ['AGE_GROUP', 'GENDER', 'RACE', 'ETHNIC', 'EDUC', 'MARSTAT']:
        if (demographic_variable in analysis_sample.columns and 
            demographic_variable in categorized_data.columns):
            
            population_distribution = analysis_sample[demographic_variable].value_counts(normalize=True).rename('population')
            sample_distribution = categorized_data[demographic_variable].value_counts(normalize=True).rename('sample')
            
            representativeness_summary[demographic_variable] = pd.concat(
                [population_distribution, sample_distribution], axis=1
            ).fillna(0).round(4).to_dict()
    
    # ==========================================================================
    # 15. DIAGNOSTICS & SENSITIVITY ANALYSIS
    # ==========================================================================
    
    diagnostic_summary = {}
    
    if CONFIG.ENABLE_MODEL_DIAGNOSTICS:
        # Local independence assessment
        local_independence = assess_local_independence(
            categorized_data, 
            [col for col in CONFIG.DIAGNOSIS_FLAG_COLUMNS if col in diagnosis_flags.columns]
        )
        save_dataframe(local_independence, get_table_path("local_independence"))
        
        diagnostic_summary["max_residual_correlation"] = float(
            np.nanmax(local_independence["MeanAbsoluteResidualCorrelation"])
        ) if not local_independence.empty else None
        
        # Entropy stability under perturbation
        entropy_stability = assess_entropy_stability(
            optimal_model, diagnosis_matrix, noise_proportion=CONFIG.DATA_PERTURBATION_RATE
        )
        with open(get_json_path("entropy_stability"), "w") as stability_file:
            json.dump(entropy_stability, stability_file, indent=2)
    
    if CONFIG.ENABLE_SENSITIVITY_ANALYSIS and comorbidity_network.number_of_nodes() > 0:
        # Community detection sensitivity
        community_sensitivity = assess_community_detection_sensitivity(
            comorbidity_network, edge_thresholds=CONFIG.NETWORK_THRESHOLDS
        )
        save_dataframe(community_sensitivity, get_table_path("community_sensitivity"))
        
        # Edge stability assessment
        edge_stability = assess_edge_stability(
            diagnosis_flags, 
            [col for col in CONFIG.DIAGNOSIS_FLAG_COLUMNS if col in diagnosis_flags.columns], 
            n_bootstrap_samples=CONFIG.BOOTSTRAP_ITERATIONS
        )
        save_dataframe(edge_stability, get_table_path("edge_stability"))
        
        # Edge stability visualization
        fig, edge_axes = plt.subplots(figsize=(7, max(3, len(edge_stability) * 0.15)))
        edge_color = get_color_palette(1, SEQUENTIAL_PALETTE)[0]
        
        edge_axes.barh(
            edge_stability["edge"], edge_stability["bootstrap_frequency"], 
            color=edge_color, edgecolor=darken(edge_color, 0.8), linewidth=0.8
        )
        edge_axes.set_xlabel("Bootstrap Inclusion Frequency")
        edge_axes.set_ylabel("Network Edge")
        edge_axes.set_title("Edge Stability Across Bootstrap Resamples")
        save_figure(os.path.splitext(get_figure_path("edge_stability", "png"))[0])
    
    # ==========================================================================
    # 16. PIPELINE SCHEMATIC & EXECUTIVE SUMMARY
    # ==========================================================================
    
    create_analysis_pipeline_schematic()
    
    # Executive summary compilation
    analysis_summary = {
        'configuration': {
            'data_file': CONFIG.DATA_FILE_PATH,
            'output_directory': CONFIG.OUTPUT_DIRECTORY,
            'random_seed': CONFIG.RANDOM_SEED,
            'inclusion_criteria': list(CONFIG.NUM_DIAGNOSES_KEEP),
            'sample_size': CONFIG.SAMPLE_SIZE_PER_GROUP,
            'edge_support_threshold': CONFIG.MINIMUM_EDGE_SUPPORT,
            'classification_confidence': CONFIG.CLASSIFICATION_CONFIDENCE_THRESHOLD,
            'diagnostics_enabled': CONFIG.ENABLE_MODEL_DIAGNOSTICS,
            'sensitivity_analysis': CONFIG.ENABLE_SENSITIVITY_ANALYSIS
        },
        'dataset': {
            'initial_sample_size': int(len(analysis_sample)),
            'final_sample_size': int(len(categorized_data))
        },
        'latent_class_analysis': {
            'optimal_components': optimal_components,
            'bic': float(model_metrics.loc[model_metrics['n_components'] == optimal_components, 'BIC'].iloc[0]),
            'aic': float(model_metrics.loc[model_metrics['n_components'] == optimal_components, 'AIC'].iloc[0]),
            'entropy': float(optimal_model['entropy']),
            'mean_max_probability': float(optimal_model['mean_max_probability'])
        },
        'classification_quality': classification_metrics,
        'validation': {
            'cross_validation': cross_validation_results,
            'bootstrap': bootstrap_results,
            'lca_network_alignment': alignment_statistics
        },
        'representativeness': representativeness_summary
    }
    
    if CONFIG.ENABLE_MODEL_DIAGNOSTICS:
        analysis_summary['diagnostics'] = {
            'maximum_residual_correlation': diagnostic_summary.get("max_residual_correlation", None)
        }
    
    with open(get_json_path("executive_summary"), 'w') as summary_file:
        json.dump(analysis_summary, summary_file, indent=2)
    
    # ==========================================================================
    # 17. FINAL DATA EXPORT
    # ==========================================================================
    
    final_dataset = optimize_dataframe_memory(categorized_data)
    final_dataset.to_csv(
        os.path.join(CONFIG.OUTPUT_DIRECTORY, "analysis_dataset.csv"), 
        index=False
    )
    
    logger.info(f"Analysis pipeline completed successfully. Outputs available in: {CONFIG.OUTPUT_DIRECTORY}")

# ==============================================================================
# 19. PIPELINE EXECUTION
# ==============================================================================

if __name__ == "__main__":
    execute_analysis_pipeline()

