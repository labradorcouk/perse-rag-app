import re
import pandas as pd
from typing import Dict, List, Any, Tuple
from .intelligent_dataframe_fixer import IntelligentDataFrameFixer

class EnhancedDataFrameCorrector:
    """
    Enhanced DataFrame corrector that combines intelligent fixing with existing correction logic.
    This system can dynamically handle any DataFrame reference issues without hardcoding specific mappings.
    """
    
    def __init__(self):
        self.intelligent_fixer = IntelligentDataFrameFixer()
        self.python_keywords = {
            'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 
            'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 
            'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 
            'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
            'print', 'type', 'isinstance', 'bool', 'tuple', 'reversed', 'filter', 
            'map', 'next', 'iter', 'slice', 'object'
        }
        
        # Common syntax error patterns and their fixes
        self.syntax_error_patterns = {
            # Incomplete method calls
            r'\.idx\s*$': '.idxmax()',  # Fix incomplete idxmax()
            r'\.idxmin\s*$': '.idxmin()',  # Fix incomplete idxmin()
            r'\.max\s*$': '.max()',  # Fix incomplete max()
            r'\.min\s*$': '.min()',  # Fix incomplete min()
            r'\.mean\s*$': '.mean()',  # Fix incomplete mean()
            r'\.sum\s*$': '.sum()',  # Fix incomplete sum()
            r'\.count\s*$': '.count()',  # Fix incomplete count()
            
            # Missing parentheses in function calls
            r'\.loc\[([^\]]+)\]\s*$': r'.loc[\1]',  # Fix incomplete loc access
            r'\.iloc\[([^\]]+)\]\s*$': r'.iloc[\1]',  # Fix incomplete iloc access
            
            # Incomplete string operations
            r'\.strip\s*$': '.strip()',  # Fix incomplete strip()
            r'\.lower\s*$': '.lower()',  # Fix incomplete lower()
            r'\.upper\s*$': '.upper()',  # Fix incomplete upper()
            
            # Incomplete DataFrame operations
            r'\.groupby\s*$': '.groupby()',  # Fix incomplete groupby()
            r'\.reset_index\s*$': '.reset_index()',  # Fix incomplete reset_index()
            r'\.drop\s*$': '.drop()',  # Fix incomplete drop()
            r'\.rename\s*$': '.rename()',  # Fix incomplete rename()
            
            # Missing closing brackets/parentheses
            r'\[([^\]]*)\s*$': r'[\1]',  # Fix missing closing bracket
            r'\(([^)]*)\s*$': r'(\1)',  # Fix missing closing parenthesis
        }
    
    def correct_dataframe_names(self, code: str, available_dataframes: Dict[str, Any] = None) -> Tuple[str, List[str]]:
        """
        Intelligently correct DataFrame names using both pattern matching and intelligent analysis.
        """
        if available_dataframes:
            self.intelligent_fixer.available_dataframes = available_dataframes
        
        # First, apply syntax error corrections
        corrected_code, syntax_fixes = self._fix_syntax_errors(code)
        
        # Then apply traditional pattern-based corrections for DataFrame references only
        corrected_code, pattern_fixes = self._apply_pattern_corrections(corrected_code, available_dataframes)
        
        # Then use the intelligent fixer only for complex DataFrame issues (not column names)
        intelligent_fixes = []
        
        # Only use intelligent fixer for actual DataFrame reference issues, not column issues
        context = self.intelligent_fixer.analyze_code_context(corrected_code)
        for issue in context['potential_issues']:
            if issue['type'] == 'undefined_dataframe':
                # This is a real DataFrame issue, let the intelligent fixer handle it
                fixed_code, fixes = self.intelligent_fixer.fix_code_dynamically(corrected_code, available_dataframes)
                intelligent_fixes.extend(fixes)
                corrected_code = fixed_code
                break  # Only fix DataFrame issues, not column issues
        
        # Combine all fixes
        all_fixes = syntax_fixes + pattern_fixes + intelligent_fixes
        
        return corrected_code, all_fixes
    
    def auto_clean_dataframe(self, df: pd.DataFrame, collection_name: str = None) -> pd.DataFrame:
        """
        Enhanced DataFrame cleaning for MongoDB data types.
        Integrates with schema configuration for better type detection.
        """
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # Try to load MongoDB schema for enhanced type detection
        schema_enhanced = False
        if collection_name:
            try:
                schema_enhanced = self._apply_schema_enhanced_cleaning(df_clean, collection_name)
            except Exception as e:
                # Fall back to automatic detection if schema loading fails
                pass
        
        # Auto-detect and fix common issues
        df_clean = self._fix_numeric_columns(df_clean)
        df_clean = self._fix_date_columns(df_clean)
        df_clean = self._fix_boolean_columns(df_clean)
        df_clean = self._fix_categorical_columns(df_clean)
        df_clean = self._fix_mongodb_specific_types(df_clean)
        
        # Special handling for battery capacity columns - force numeric conversion
        df_clean = self._force_battery_capacity_conversion(df_clean)
        
        # Final aggressive cleaning for any remaining object columns that should be numeric
        df_clean = self._aggressive_numeric_cleaning(df_clean)
        
        return df_clean
    
    def _apply_schema_enhanced_cleaning(self, df: pd.DataFrame, collection_name: str) -> bool:
        """
        Apply schema-enhanced cleaning using MongoDB schema configuration.
        Returns True if schema was successfully applied.
        """
        try:
            # Try to import and use MongoDB schema manager
            from .mongodb_schema_manager import MongoDBSchemaManager
            
            # Initialize schema manager
            schema_manager = MongoDBSchemaManager()
            
            # Get schema for this collection
            schema = schema_manager.get_collection_schema(collection_name)
            if not schema:
                return False
            
            # Apply schema-based type hints
            if 'column_types' in schema:
                for col, expected_type in schema['column_types'].items():
                    if col in df.columns:
                        df[col] = self._convert_column_by_schema_hint(df[col], expected_type)
            
            return True
            
        except ImportError:
            # Schema manager not available
            return False
        except Exception:
            # Any other error
            return False
    
    def _convert_column_by_schema_hint(self, series: pd.Series, expected_type: str) -> pd.Series:
        """
        Convert a column based on schema type hints.
        """
        try:
            if expected_type.lower() in ['numeric', 'float', 'int', 'number']:
                return pd.to_numeric(series, errors='coerce')
            elif expected_type.lower() in ['date', 'datetime', 'timestamp']:
                return pd.to_datetime(series, errors='coerce')
            elif expected_type.lower() in ['boolean', 'bool']:
                return series.map({'Yes': True, 'No': False, '1': True, '0': False}).fillna(series)
            elif expected_type.lower() in ['categorical', 'category', 'enum']:
                return series.astype('category')
            else:
                return series
        except:
            return series
    
    def _fix_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and fix numeric columns with mixed data types."""
        for col in df.columns:
            if df[col].dtype == 'object':  # Only process object columns
                # Check if column should be numeric
                if self._should_be_numeric(col, df[col]):
                    df[col] = self._smart_numeric_conversion(df[col])
        
        return df
    
    def _should_be_numeric(self, column_name: str, series: pd.Series) -> bool:
        """
        Enhanced numeric detection for MongoDB data types.
        Uses dynamic patterns and adaptive thresholds.
        """
        # MongoDB-specific numeric patterns
        numeric_patterns = [
            # Energy and power
            r'^\d+\.?\d*\s*[kK][wW][hH]$',  # kWh values
            r'^\d+\.?\d*\s*[wW]$',          # Watt values
            r'^\d+\.?\d*\s*[kK][wW]$',      # kW values
            
            # Distance and range
            r'^\d+\.?\d*\s*[kK][mM]$',      # km values
            r'^\d+\.?\d*\s*[mM][iI]$',      # miles values
            r'^\d+\.?\d*\s*[mM][pP][gG]$',  # MPG values
            
            # Consumption and efficiency
            r'^\d+\.?\d*\s*[lL]/100[kmKm]$',  # L/100km values
            r'^\d+\.?\d*\s*[kK][wW][hH]/100[kmKm]$',  # kWh/100km values
            
            # Standard numbers
            r'^\d+\.?\d*$',                  # Standard numbers
            r'^\d+$',                        # Integers
        ]
        
        # Enhanced keyword detection with MongoDB context
        numeric_keywords = [
            # Vehicle-specific
            'capacity', 'efficiency', 'range', 'consumption', 'economy',
            'battery', 'voltage', 'current', 'speed', 'weight', 'length', 
            'width', 'height', 'volume', 'area', 'price', 'cost', 'rate',
            'ratio', 'percentage', 'score', 'rating', 'index', 'factor',
            'coefficient', 'metric', 'power', 'energy',
            
            # MongoDB-specific patterns
            'wlpt', 'nedc', 'real', 'estimate', 'teh', 'fuel', 'co2',
            'available', 'status', 'count', 'total', 'average', 'mean',
            'max', 'min', 'sum', 'difference', 'gap', 'margin'
        ]
        
        # Check column name patterns (case-insensitive)
        column_lower = column_name.lower()
        if any(keyword in column_lower for keyword in numeric_keywords):
            return True
        
        # Check for MongoDB ObjectId patterns (should NOT be numeric)
        if 'id' in column_lower and any(id_pattern in column_lower for id_pattern in ['object', 'mongo', 'db']):
            return False
        
        # Enhanced content pattern analysis
        sample_values = series.dropna().head(200)  # Increased sample size
        if len(sample_values) == 0:
            return False
        
        numeric_count = 0
        total_count = 0
        unit_count = 0
        
        for value in sample_values:
            if pd.isna(value):
                continue
            
            value_str = str(value).strip()
            if value_str in ['', 'N/A', 'NA', 'None', 'Unknown', 'TBD', 'n/a', 'na']:
                continue
            
            # Check against numeric patterns
            if any(re.match(pattern, value_str) for pattern in numeric_patterns):
                numeric_count += 1
                if any(unit in value_str.lower() for unit in ['kwh', 'km', 'mi', 'mpg', 'l/100']):
                    unit_count += 1
            else:
                # Try basic numeric conversion
                try:
                    float(value_str)
                    numeric_count += 1
                except:
                    pass
            
            total_count += 1
        
        # Adaptive threshold based on data characteristics
        if total_count == 0:
            return False
        
        numeric_ratio = numeric_count / total_count
        unit_ratio = unit_count / total_count if total_count > 0 else 0
        
        # If high unit presence, lower threshold (units indicate numeric intent)
        if unit_ratio > 0.3:
            threshold = 0.5  # Lower threshold for unit-heavy data
        elif unit_ratio > 0.1:
            threshold = 0.6  # Medium threshold for some units
        else:
            threshold = 0.7  # Standard threshold for no units
        
        return numeric_ratio > threshold
    
    def _smart_numeric_conversion(self, series: pd.Series) -> pd.Series:
        """
        Enhanced numeric conversion for MongoDB data types.
        Handles units, ranges, and complex formats.
        """
        def parse_value(value):
            if pd.isna(value):
                return pd.NA
            
            value_str = str(value).strip()
            
            # Handle MongoDB-specific non-numeric values
            if value_str in ['', 'N/A', 'NA', 'None', 'Unknown', 'TBD', 'n/a', 'na', 'NULL', 'null']:
                return pd.NA
            
            # Handle percentage values
            if '%' in value_str:
                try:
                    return float(value_str.replace('%', '').strip()) / 100
                except:
                    return pd.NA
            
            # Handle range values (take average)
            if '-' in value_str or ' to ' in value_str:
                try:
                    if '-' in value_str:
                        parts = value_str.split('-')
                    else:
                        parts = value_str.split(' to ')
                    
                    if len(parts) == 2:
                        # Extract numeric parts, handling units
                        start_str = re.sub(r'[^\d.]', '', parts[0])
                        end_str = re.sub(r'[^\d.]', '', parts[1])
                        
                        if start_str and end_str:
                            start = float(start_str)
                            end = float(end_str)
                            return (start + end) / 2
                except:
                    pass
            
            # Handle units - extract numeric part
            try:
                # Common MongoDB units
                unit_patterns = [
                    r'^(\d+\.?\d*)\s*[kK][wW][hH]$',      # kWh
                    r'^(\d+\.?\d*)\s*[wW]$',               # W
                    r'^(\d+\.?\d*)\s*[kK][wW]$',           # kW
                    r'^(\d+\.?\d*)\s*[kK][mM]$',           # km
                    r'^(\d+\.?\d*)\s*[mM][iI]$',           # miles
                    r'^(\d+\.?\d*)\s*[mM][pP][gG]$',       # MPG
                    r'^(\d+\.?\d*)\s*[lL]/100[kmKm]$',     # L/100km
                    r'^(\d+\.?\d*)\s*[kK][wW][hH]/100[kmKm]$',  # kWh/100km
                ]
                
                for pattern in unit_patterns:
                    match = re.match(pattern, value_str)
                    if match:
                        return float(match.group(1))
                
                # Try to extract numeric part from mixed strings
                numeric_part = re.sub(r'[^\d.]', '', value_str)
                if numeric_part:
                    return float(numeric_part)
                    
            except:
                pass
            
            # Handle MongoDB ObjectId (should not be numeric)
            if re.match(r'^[0-9a-fA-F]{24}$', value_str):
                return pd.NA
            
            # Standard numeric conversion
            try:
                return float(value_str)
            except:
                return pd.NA
        
        return series.apply(parse_value)
    
    def _fix_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced date detection for MongoDB data types.
        Handles ISODate, ObjectId dates, and various formats.
        """
        # MongoDB-specific date patterns
        date_keywords = [
            'date', 'time', 'created', 'updated', 'launch', 'release',
            'from', 'to', 'start', 'end', 'modified', 'timestamp',
            'available', 'published', 'expired', 'valid', 'effective'
        ]
        
        # MongoDB date patterns
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}',           # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}',           # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}',           # DD-MM-YYYY
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # Space format
        ]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                column_lower = col.lower()
                
                # Check column name patterns
                is_date_column = any(keyword in column_lower for keyword in date_keywords)
                
                # Check content patterns
                sample_values = df[col].dropna().head(50)
                date_like_count = 0
                
                for value in sample_values:
                    value_str = str(value).strip()
                    if any(re.match(pattern, value_str) for pattern in date_patterns):
                        date_like_count += 1
                
                # If column name suggests date OR content looks date-like, convert
                if is_date_column or (len(sample_values) > 0 and date_like_count / len(sample_values) > 0.3):
                    try:
                        # Try multiple date parsing strategies
                        df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                        
                        # If conversion resulted in mostly NaT, try alternative formats
                        if df[col].isna().sum() > len(df[col]) * 0.7:
                            # Try parsing as Unix timestamp
                            try:
                                df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')
                            except:
                                pass
                    except:
                        pass  # Keep as is if conversion fails
        
        return df
    
    def _fix_boolean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and fix boolean columns."""
        boolean_keywords = ['status', 'active', 'enabled', 'available', 'is_', 'has_']
        
        for col in df.columns:
            if df[col].dtype == 'object':
                column_lower = col.lower()
                if any(keyword in column_lower for keyword in boolean_keywords):
                    # Check if column has boolean-like values
                    unique_values = df[col].dropna().unique()
                    if len(unique_values) <= 5:  # Likely categorical/boolean
                        # Convert common boolean patterns
                        df[col] = df[col].map({
                            'Yes': True, 'No': False,
                            'True': True, 'False': False,
                            '1': True, '0': False,
                            'Active': True, 'Inactive': False,
                            'Available': True, 'Unavailable': False
                        }).fillna(df[col])  # Keep original if no mapping
        
        return df
    
    def _fix_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced categorical detection for MongoDB data types.
        Handles enums, status fields, and low-cardinality data.
        """
        # MongoDB-specific categorical patterns
        categorical_keywords = [
            'status', 'type', 'category', 'brand', 'make', 'model', 'version',
            'color', 'fuel', 'transmission', 'drive', 'body', 'segment',
            'level', 'grade', 'class', 'rating', 'tier', 'group', 'family'
        ]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                column_lower = col.lower()
                
                # Check if column name suggests categorical
                is_categorical_name = any(keyword in column_lower for keyword in categorical_keywords)
                
                # Analyze content patterns
                unique_count = df[col].nunique()
                total_count = len(df[col])
                
                # Calculate cardinality ratio
                if total_count > 0:
                    cardinality_ratio = unique_count / total_count
                    
                    # Adaptive threshold based on column characteristics
                    if is_categorical_name:
                        threshold = 0.3  # Lower threshold for name-suggested categoricals
                    elif 'id' in column_lower and unique_count < 100:
                        threshold = 0.5  # Medium threshold for ID-like fields
                    else:
                        threshold = 0.1  # Standard threshold for general fields
                    
                    # Check if column should be categorical
                    if cardinality_ratio < threshold and unique_count > 0:
                        # Additional validation: check if values look categorical
                        sample_values = df[col].dropna().head(100)
                        categorical_like = 0
                        
                        for value in sample_values:
                            value_str = str(value).strip()
                            # Check if value looks like a category (not numeric, not too long)
                            if (len(value_str) < 50 and 
                                not value_str.replace('.', '').replace('-', '').isdigit() and
                                value_str not in ['', 'N/A', 'NA', 'None']):
                                categorical_like += 1
                        
                        # If most values look categorical, convert
                        if len(sample_values) > 0 and categorical_like / len(sample_values) > 0.7:
                            try:
                                df[col] = df[col].astype('category')
                            except:
                                pass  # Keep as is if conversion fails
        
        return df
    
    def _fix_mongodb_specific_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle MongoDB-specific data types like ObjectId, embedded documents, and arrays.
        """
        for col in df.columns:
            if df[col].dtype == 'object':
                column_lower = col.lower()
                
                # Handle ObjectId columns (MongoDB's primary key)
                if 'id' in column_lower and ('object' in column_lower or 'mongo' in column_lower):
                    # Keep ObjectId as string - don't convert to numeric
                    continue
                
                # Handle embedded document columns
                if 'embedding' in column_lower:
                    # Keep embedding vectors as is
                    continue
                
                # Handle array columns (common in MongoDB)
                sample_values = df[col].dropna().head(20)
                array_like_count = 0
                
                for value in sample_values:
                    value_str = str(value)
                    # Check if value looks like an array or list
                    if (value_str.startswith('[') and value_str.endswith(']')) or \
                       (value_str.startswith('{') and value_str.endswith('}')) or \
                       (',' in value_str and len(value_str) > 20):
                        array_like_count += 1
                
                # If column looks like it contains arrays/objects, keep as object
                if len(sample_values) > 0 and array_like_count / len(sample_values) > 0.5:
                    continue
                
                # Handle mixed type columns (common in MongoDB)
                type_counts = {}
                for value in sample_values:
                    if pd.isna(value):
                        continue
                    value_type = type(value).__name__
                    type_counts[value_type] = type_counts.get(value_type, 0) + 1
                
                # If column has mixed types, try to normalize
                if len(type_counts) > 2:  # More than 2 different types
                    # Try to convert everything to string for consistency
                    try:
                        df[col] = df[col].astype(str)
                    except:
                        pass
        
        return df
    
    def _force_battery_capacity_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Force numeric conversion for battery capacity columns.
        This is a special case because battery capacity is critical for analysis.
        """
        battery_columns = [col for col in df.columns if 'battery' in col.lower() and 'capacity' in col.lower()]
        
        for col in battery_columns:
            if df[col].dtype == 'object':
                print(f"🔋 Force-converting battery column: {col}")
                
                # First, try aggressive cleaning
                cleaned_series = df[col].copy()
                
                # Handle common MongoDB data issues
                cleaned_series = cleaned_series.replace([
                    'N/A', 'NA', 'None', 'Unknown', 'TBD', 'n/a', 'na', 'NULL', 'null',
                    '', ' ', 'nan', 'NaN', 'NAN', 'inf', 'Inf', 'INF', '-inf', '-Inf', '-INF'
                ], pd.NA)
                
                # Pre-process values with ellipsis to extract numeric part
                def preprocess_ellipsis(value):
                    if pd.isna(value):
                        return value
                    
                    value_str = str(value).strip()
                    if '...' in value_str:
                        # Extract the numeric part before ellipsis
                        before_ellipsis = value_str.split('...')[0]
                        # Try to extract just the numeric part
                        numeric_part = re.sub(r'[^\d.]', '', before_ellipsis)
                        if numeric_part and numeric_part != '.':
                            return numeric_part
                    
                    return value
                
                # Apply pre-processing
                cleaned_series = cleaned_series.apply(preprocess_ellipsis)
                
                # Try to extract numeric values from various formats
                def extract_battery_value(value):
                    if pd.isna(value):
                        return pd.NA
                    
                    value_str = str(value).strip()
                    
                    # Handle empty/NA values
                    if value_str in ['', 'N/A', 'NA', 'None', 'Unknown', 'TBD', 'n/a', 'na', 'NULL', 'null']:
                        return pd.NA
                    
                    # Handle percentage values
                    if '%' in value_str:
                        try:
                            return float(value_str.replace('%', '').strip()) / 100
                        except:
                            return pd.NA
                    
                    # Handle range values (take average)
                    if '-' in value_str or ' to ' in value_str:
                        try:
                            if '-' in value_str:
                                parts = value_str.split('-')
                            else:
                                parts = value_str.split(' to ')
                            
                            if len(parts) == 2:
                                # Extract numeric parts, handling units
                                start_str = re.sub(r'[^\d.]', '', parts[0])
                                end_str = re.sub(r'[^\d.]', '', parts[1])
                                
                                if start_str and end_str:
                                    start = float(start_str)
                                    end = float(end_str)
                                    return (start + end) / 2
                        except:
                            pass
                    
                    # Handle units - extract numeric part
                    try:
                        # Common battery capacity units
                        unit_patterns = [
                            r'^(\d+\.?\d*)\s*[kK][wW][hH]$',      # kWh
                            r'^(\d+\.?\d*)\s*[wW][hH]$',           # Wh
                            r'^(\d+\.?\d*)\s*[aA][hH]$',           # Ah
                            r'^(\d+\.?\d*)\s*[mM][aA][hH]$',       # mAh
                            r'^(\d+\.?\d*)\s*[kK][aA][hH]$',       # kAh
                        ]
                        
                        for pattern in unit_patterns:
                            match = re.match(pattern, value_str)
                            if match:
                                return float(match.group(1))
                        
                        # Try to extract numeric part from mixed strings
                        numeric_part = re.sub(r'[^\d.]', '', value_str)
                        if numeric_part:
                            return float(numeric_part)
                        
                        # Handle ellipsis and truncated values (common in MongoDB)
                        if '...' in value_str:
                            # Extract the numeric part before ellipsis
                            before_ellipsis = value_str.split('...')[0]
                            numeric_part = re.sub(r'[^\d.]', '', before_ellipsis)
                            if numeric_part:
                                return float(numeric_part)
                        
                        # Handle values with dots but no clear units
                        if '.' in value_str:
                            # Try to extract just the numeric part
                            parts = value_str.split('.')
                            if len(parts) >= 2:
                                # Check if first part is numeric
                                if parts[0].isdigit():
                                    try:
                                        return float(parts[0] + '.' + parts[1])
                                    except:
                                        pass
                        
                        # Handle values that start with numbers but have trailing text
                        if value_str and value_str[0].isdigit():
                            # Try to extract the numeric part from the beginning
                            numeric_chars = []
                            for char in value_str:
                                if char.isdigit() or char == '.':
                                    numeric_chars.append(char)
                                else:
                                    break
                            
                            if numeric_chars:
                                try:
                                    return float(''.join(numeric_chars))
                                except:
                                    pass
                            
                    except:
                        pass
                    
                    # Handle MongoDB ObjectId (should not be numeric)
                    if re.match(r'^[0-9a-fA-F]{24}$', value_str):
                        return pd.NA
                    
                    # Standard numeric conversion
                    try:
                        return float(value_str)
                    except:
                        # Final fallback: try to extract any numeric content
                        try:
                            # Remove all non-numeric characters except dots
                            cleaned = re.sub(r'[^\d.]', '', value_str)
                            if cleaned and cleaned != '.':
                                # Handle cases where we have multiple dots
                                if cleaned.count('.') > 1:
                                    # Take the first two parts
                                    parts = cleaned.split('.')
                                    if len(parts) >= 2:
                                        cleaned = parts[0] + '.' + parts[1]
                                return float(cleaned)
                        except:
                            pass
                        
                        return pd.NA
                
                # Apply the conversion with debugging
                print(f"🔍 Converting {col} - sample values before conversion:")
                print(f"   Original: {cleaned_series.head(3).tolist()}")
                
                df[col] = cleaned_series.apply(extract_battery_value)
                
                print(f"   After conversion: {df[col].head(3).tolist()}")
                print(f"   Data type: {df[col].dtype}")
                print(f"   Non-null count: {df[col].count()}/{len(df[col])}")
                
                # Verify conversion worked
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    print(f"✅ Successfully converted {col} to {df[col].dtype}")
                    print(f"   Sample values: {df[col].dropna().head(3).tolist()}")
                else:
                    print(f"⚠️ Failed to convert {col} - still {df[col].dtype}")
                    print(f"   Sample values: {df[col].head(3).tolist()}")
        
        return df
    
    def _aggressive_numeric_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final aggressive cleaning for any remaining object columns that should be numeric.
        This catches edge cases that the other methods might miss.
        """
        # Look for columns that might be numeric but weren't caught
        potential_numeric_columns = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column name suggests numeric
                col_lower = col.lower()
                numeric_indicators = [
                    'capacity', 'efficiency', 'range', 'consumption', 'economy',
                    'battery', 'voltage', 'current', 'speed', 'weight', 'length', 
                    'width', 'height', 'volume', 'area', 'price', 'cost', 'rate',
                    'ratio', 'percentage', 'score', 'rating', 'index', 'factor',
                    'coefficient', 'metric', 'power', 'energy', 'wlpt', 'nedc',
                    'real', 'estimate', 'teh', 'fuel', 'co2'
                ]
                
                if any(indicator in col_lower for indicator in numeric_indicators):
                    # Check content
                    sample_values = df[col].dropna().head(50)
                    if len(sample_values) > 0:
                        numeric_like = 0
                        for value in sample_values:
                            try:
                                # Try to extract numeric part
                                value_str = str(value).strip()
                                # Remove common units and try conversion
                                cleaned_value = re.sub(r'[^\d.]', '', value_str)
                                if cleaned_value and float(cleaned_value) > 0:
                                    numeric_like += 1
                            except:
                                pass
                        
                        if numeric_like > 0:
                            potential_numeric_columns.append((col, numeric_like / len(sample_values)))
        
        # Sort by confidence and process
        potential_numeric_columns.sort(key=lambda x: x[1], reverse=True)
        
        for col, confidence in potential_numeric_columns:
            if confidence > 0.3:  # If more than 30% look numeric
                print(f"🔧 Aggressive cleaning for {col} (confidence: {confidence:.2f})")
                
                try:
                    # Try to convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # If successful, show results
                    if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                        print(f"✅ Aggressively converted {col} to {df[col].dtype}")
                        print(f"   Non-null values: {df[col].count()}/{len(df[col])}")
                    else:
                        print(f"⚠️ Aggressive conversion failed for {col}")
                except Exception as e:
                    print(f"❌ Error in aggressive conversion for {col}: {e}")
        
        return df
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced data quality report for MongoDB data types.
        Provides detailed insights into data structure and quality.
        """
        try:
            report = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns_processed': [],
                'data_type_changes': [],
                'missing_values': {},
                'mongodb_specific_issues': {},
                'quality_score': 0.0,
                'data_type_distribution': {},
                'recommendations': []
            }
            
            for col in df.columns:
                try:
                    current_dtype = df[col].dtype
                    missing_count = df[col].isna().sum()
                    missing_percentage = (missing_count / len(df)) * 100
                    
                    # Track data type distribution
                    dtype_name = str(current_dtype)
                    report['data_type_distribution'][dtype_name] = report['data_type_distribution'].get(dtype_name, 0) + 1
                    
                    # Analyze missing values
                    report['missing_values'][col] = {
                        'count': missing_count,
                        'percentage': missing_percentage,
                        'dtype': dtype_name
                    }
                    
                    # MongoDB-specific analysis (skip if column contains arrays)
                    if self._contains_mongodb_arrays(df[col]):
                        report['mongodb_specific_issues'][col] = {
                            'type': 'mongodb_array',
                            'message': 'Column contains MongoDB array fields - skipping detailed analysis',
                            'sample_lengths': self._get_array_lengths(df[col])
                        }
                        # Skip quality score calculation for array columns
                        continue
                    else:
                        col_analysis = self._analyze_column_for_mongodb_issues(df[col], col)
                        if col_analysis:
                            report['mongodb_specific_issues'][col] = col_analysis
                    
                    # Calculate quality score (lower missing values = higher score)
                    report['quality_score'] += (1 - missing_percentage / 100)
                    
                    # Generate recommendations
                    if missing_percentage > 50:
                        report['recommendations'].append(f"Column '{col}' has {missing_percentage:.1f}% missing values - consider investigation")
                    
                    if current_dtype == 'object' and missing_percentage < 20:
                        # Check if object column might be better as another type
                        try:
                            unique_ratio = df[col].nunique() / len(df[col])
                            if unique_ratio < 0.1:
                                report['recommendations'].append(f"Column '{col}' has low cardinality ({unique_ratio:.2f}) - consider categorical type")
                        except Exception as e:
                            # Skip if there are issues with unique calculation (e.g., unhashable types)
                            report['recommendations'].append(f"Column '{col}' - unable to calculate cardinality due to data type issues")
                    
                except Exception as col_error:
                    # Handle individual column errors gracefully
                    report['mongodb_specific_issues'][col] = {
                        'type': 'error',
                        'message': f'Error analyzing column: {str(col_error)}',
                        'error_type': type(col_error).__name__
                    }
                    continue
            
            # Calculate final quality score (only for non-array columns)
            non_array_columns = [col for col in df.columns if not self._contains_mongodb_arrays(df[col])]
            if non_array_columns:
                report['quality_score'] = report['quality_score'] / len(non_array_columns)
            else:
                report['quality_score'] = 0.0
            
            # Overall recommendations
            if report['quality_score'] < 0.7:
                report['recommendations'].append("Overall data quality is low - consider data cleaning pipeline")
            
            # Special analysis for battery capacity columns (skip array columns)
            battery_columns = [col for col in df.columns if 'battery' in col.lower() and 'capacity' in col.lower() and not self._contains_mongodb_arrays(df[col])]
            if battery_columns:
                report['battery_capacity_analysis'] = {}
                for col in battery_columns:
                    try:
                        col_analysis = {
                            'dtype': str(df[col].dtype),
                            'total_values': len(df[col]),
                            'non_null_values': df[col].count(),
                            'null_percentage': (df[col].isna().sum() / len(df[col])) * 100,
                            'sample_values': self._safe_sample_values(df[col], 5),
                            'unique_values': df[col].nunique()
                        }
                        
                        # Check if column is numeric
                        if df[col].dtype in ['object', 'string']:
                            # Try to detect if it should be numeric
                            sample_values = self._safe_sample_values(df[col], 20)
                            numeric_like = 0
                            for value in sample_values:
                                try:
                                    float(str(value).replace('kWh', '').replace('Wh', '').replace('Ah', '').strip())
                                    numeric_like += 1
                                except:
                                    pass
                            
                            col_analysis['numeric_like_percentage'] = (numeric_like / len(sample_values)) * 100 if len(sample_values) > 0 else 0
                            col_analysis['should_be_numeric'] = col_analysis['numeric_like_percentage'] > 50
                        
                        report['battery_capacity_analysis'][col] = col_analysis
                    except Exception as e:
                        report['battery_capacity_analysis'][col] = {
                            'error': f'Unable to analyze battery column: {str(e)}'
                        }
            
            return report
            
        except Exception as e:
            # Fallback report if main analysis fails
            return {
                'error': f'Data quality report generation failed: {str(e)}',
                'error_type': type(e).__name__,
                'total_rows': len(df) if 'df' in locals() else 0,
                'total_columns': len(df.columns) if 'df' in locals() else 0,
                'recommendations': ['Data quality analysis encountered errors - consider manual inspection']
            }
    
    def _contains_mongodb_arrays(self, series: pd.Series) -> bool:
        """
        Check if a series contains MongoDB array fields.
        """
        try:
            # Sample a few values to check for arrays
            sample_values = series.dropna().head(10)
            for value in sample_values:
                if isinstance(value, list):
                    return True
            return False
        except:
            return False
    
    def _get_array_lengths(self, series: pd.Series) -> Dict[str, Any]:
        """
        Get information about array lengths in a series.
        """
        try:
            sample_values = series.dropna().head(10)
            lengths = []
            for value in sample_values:
                if isinstance(value, list):
                    lengths.append(len(value))
            
            if lengths:
                return {
                    'min_length': min(lengths),
                    'max_length': max(lengths),
                    'avg_length': sum(lengths) / len(lengths),
                    'sample_lengths': lengths[:5]  # First 5 lengths
                }
            else:
                return {'message': 'No arrays found in sample'}
        except:
            return {'error': 'Unable to analyze array lengths'}
    
    def _safe_sample_values(self, series: pd.Series, n: int) -> List[Any]:
        """
        Safely get sample values from a series, handling unhashable types.
        """
        try:
            sample = series.dropna().head(n)
            # Convert to list safely, handling unhashable types
            values = []
            for value in sample:
                try:
                    # Try to convert to string representation if it's unhashable
                    if isinstance(value, (list, dict, set)):
                        values.append(str(value)[:100])  # Truncate long representations
                    else:
                        values.append(value)
                except:
                    values.append(f"<unhashable_type_{type(value).__name__}>")
            return values
        except:
            return [f"<error_reading_values>"]
    
    def _analyze_column_for_mongodb_issues(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Analyze a column for MongoDB-specific data quality issues.
        """
        analysis = {}
        column_lower = column_name.lower()
        
        # Check for ObjectId patterns
        if 'id' in column_lower:
            sample_values = series.dropna().head(100)
            objectid_count = 0
            
            for value in sample_values:
                if re.match(r'^[0-9a-fA-F]{24}$', str(value)):
                    objectid_count += 1
            
            if objectid_count > 0:
                analysis['objectid_detected'] = True
                analysis['objectid_percentage'] = (objectid_count / len(sample_values)) * 100
        
        # Check for embedded documents
        if any(keyword in column_lower for keyword in ['embedding', 'document', 'nested']):
            sample_values = series.dropna().head(50)
            embedded_count = 0
            
            for value in sample_values:
                value_str = str(value)
                if (value_str.startswith('[') and value_str.endswith(']')) or \
                   (value_str.startswith('{') and value_str.endswith('}')):
                    embedded_count += 1
            
            if embedded_count > 0:
                analysis['embedded_documents'] = True
                analysis['embedded_percentage'] = (embedded_count / len(sample_values)) * 100
        
        # Check for mixed data types
        type_counts = {}
        for value in series.dropna().head(100):
            value_type = type(value).__name__
            type_counts[value_type] = type_counts.get(value_type, 0) + 1
        
        if len(type_counts) > 2:
            analysis['mixed_types'] = True
            analysis['type_distribution'] = type_counts
        
        return analysis

    def _fix_syntax_errors(self, code: str) -> Tuple[str, List[str]]:
        """
        Dynamically fix common syntax errors in the code.
        This is non-aggressive and only fixes obvious syntax issues.
        """
        fixes = []
        corrected_code = code
        
        # Split code into lines to process each line separately
        lines = corrected_code.split('\n')
        corrected_lines = []
        
        for i, line in enumerate(lines):
            original_line = line
            corrected_line = line
            
            # Apply syntax error patterns to each line
            for pattern, replacement in self.syntax_error_patterns.items():
                if re.search(pattern, corrected_line):
                    corrected_line = re.sub(pattern, replacement, corrected_line)
                    if corrected_line != original_line:
                        fixes.append(f"Fixed syntax error on line {i+1}: {original_line.strip()} → {corrected_line.strip()}")
            
            corrected_lines.append(corrected_line)
        
        corrected_code = '\n'.join(corrected_lines)
        
        return corrected_code, fixes
    
    def _apply_pattern_corrections(self, code: str, available_dataframes: Dict[str, Any] = None) -> Tuple[str, List[str]]:
        """
        Apply traditional pattern-based corrections as a fallback.
        """
        fixes = []
        
        if not available_dataframes:
            return code, fixes
        
        # Find potential DataFrame references
        potential_dfs = self._find_potential_dataframe_references(code)
        
        # Get preferred DataFrame name
        preferred_df = self._get_preferred_dataframe(available_dataframes)
        
        # Replace potential DataFrame references
        for potential_df in potential_dfs:
            if potential_df not in available_dataframes and potential_df not in self.python_keywords:
                # Check if it looks like a DataFrame reference
                if self._looks_like_dataframe_reference(potential_df, code):
                    code = re.sub(rf'\b{potential_df}\b', preferred_df, code)
                    fixes.append(f"Replaced '{potential_df}' with '{preferred_df}'")
        
        return code, fixes
    
    def _find_potential_dataframe_references(self, code: str) -> List[str]:
        """
        Find potential DataFrame references in the code.
        """
        # Look for variable names that might be DataFrames
        variable_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        potential_refs = re.findall(variable_pattern, code)
        
        # Filter out Python keywords and common non-DataFrame names
        filtered_refs = []
        for ref in potential_refs:
            if (ref not in self.python_keywords and 
                ref not in ['pd', 'np', 'plt', 'df', 'data', 'result', 'output'] and
                len(ref) > 2):  # Likely not a DataFrame if too short
                filtered_refs.append(ref)
        
        return list(set(filtered_refs))
    
    def _looks_like_dataframe_reference(self, potential_df: str, code: str) -> bool:
        """
        Determine if a potential reference looks like a DataFrame usage.
        """
        # First, check if it's likely a column name (starts with capital letter or contains underscore)
        if potential_df[0].isupper() or '_' in potential_df:
            # This is likely a column name, not a DataFrame
            return False
        
        # Check if it's a common method name
        common_methods = {
            'loc', 'iloc', 'head', 'tail', 'describe', 'info', 'columns', 'shape',
            'max', 'min', 'mean', 'sum', 'count', 'dropna', 'fillna', 'sort_values',
            'groupby', 'merge', 'join', 'concat', 'append', 'drop', 'rename', 'copy',
            'astype', 'dtypes', 'index', 'idxmax', 'idxmin', 'nlargest', 'nsmallest'
        }
        if potential_df in common_methods:
            return False
        
        # Check if it's a common variable name that's not a DataFrame
        common_vars = {
            'result', 'fig', 'data', 'output', 'df', 'df1', 'df2', 'df3', 'df4', 'df5',
            'pd', 'np', 'plt', 'matplotlib', 'seaborn', 'sns'
        }
        if potential_df in common_vars:
            return False
        
        # Check if it's part of a larger column name (e.g., "Battery" in "Battery_Capacity_Full")
        # Look for patterns like df['Column_Name'] or df.Column_Name
        column_patterns = [
            rf"df\d*\['{potential_df}[^']*'\]",  # df['Battery_Capacity_Full']
            rf"df\d*\.{potential_df}[^'\"\s]*",  # df.Battery_Capacity_Full
            rf"df\d*\[{potential_df}[^'\]]*\]",  # df[Battery_Capacity_Full]
        ]
        
        for pattern in column_patterns:
            if re.search(pattern, code):
                # This is part of a column name, not a DataFrame
                return False
        
        # Look for DataFrame-like operations - more specific patterns
        # Only consider it a DataFrame if it's followed by specific DataFrame operations
        df_operations = [
            rf'\b{potential_df}\[',  # Column access with word boundary
            rf'\b{potential_df}\.head\b',  # Common DataFrame methods with word boundaries
            rf'\b{potential_df}\.tail\b',
            rf'\b{potential_df}\.describe\b',
            rf'\b{potential_df}\.info\b',
            rf'\b{potential_df}\.columns\b',
            rf'\b{potential_df}\.shape\b',
            rf'\b{potential_df}\.loc\b',
            rf'\b{potential_df}\.iloc\b',
            rf'\b{potential_df}\.max\b',
            rf'\b{potential_df}\.min\b',
            rf'\b{potential_df}\.mean\b',
            rf'\b{potential_df}\.sum\b',
            rf'\b{potential_df}\.count\b',
            rf'\b{potential_df}\.dropna\b',
            rf'\b{potential_df}\.fillna\b',
            rf'\b{potential_df}\.sort_values\b',
            rf'\b{potential_df}\.groupby\b',
            rf'\b{potential_df}\.merge\b',
            rf'\b{potential_df}\.join\b',
            rf'\b{potential_df}\.concat\b',
            rf'\b{potential_df}\.append\b',
            rf'\b{potential_df}\.drop\b',
            rf'\b{potential_df}\.rename\b',
            rf'\b{potential_df}\.copy\b',
            rf'\b{potential_df}\.astype\b',
            rf'\b{potential_df}\.dtypes\b',
            rf'\b{potential_df}\.index\b',
            rf'\b{potential_df}\.idxmax\b',
            rf'\b{potential_df}\.idxmin\b',
        ]
        
        for pattern in df_operations:
            if re.search(pattern, code):
                return True
        
        return False
    
    def _get_preferred_dataframe(self, available_dataframes: Dict[str, Any]) -> str:
        """
        Get the preferred DataFrame name from available DataFrames.
        """
        if not available_dataframes:
            return 'df1'
        
        # Prefer 'df1' if available
        if 'df1' in available_dataframes:
            return 'df1'
        
        # Otherwise, return the first available DataFrame
        return list(available_dataframes.keys())[0]
    
    def validate_dataframe_references(self, code: str, available_dataframes: Dict[str, Any] = None) -> List[str]:
        """
        Validate DataFrame references and return any unresolved issues.
        """
        if available_dataframes:
            self.intelligent_fixer.available_dataframes = available_dataframes
        
        context = self.intelligent_fixer.analyze_code_context(code)
        unresolved_issues = []
        
        for issue in context['potential_issues']:
            if issue['type'] == 'undefined_dataframe':
                unresolved_issues.append(f"Undefined DataFrame: {issue['dataframe']}")
            elif issue['type'] == 'undefined_column':
                unresolved_issues.append(f"Undefined column: {issue['column']}")
            elif issue['type'] == 'incorrect_method':
                unresolved_issues.append(f"Incorrect method: {issue['object']}.{issue['method']}")
        
        return unresolved_issues
    
    def get_dataframe_info(self, available_dataframes: Dict[str, Any] = None) -> str:
        """
        Get comprehensive information about available DataFrames for LLM context.
        """
        if not available_dataframes:
            return "No DataFrames available"
        
        info_parts = []
        
        for df_name, df in available_dataframes.items():
            if hasattr(df, 'columns'):
                info_parts.append(f"DataFrame '{df_name}': {len(df.columns)} columns")
                info_parts.append(f"  Sample columns: {list(df.columns)[:5]}")
                if hasattr(df, 'shape'):
                    info_parts.append(f"  Shape: {df.shape}")
            else:
                info_parts.append(f"DataFrame '{df_name}': No column information available")
        
        return "\n".join(info_parts)
    
    def generate_smart_suggestions(self, code: str, available_dataframes: Dict[str, Any] = None) -> List[str]:
        """
        Generate smart suggestions for improving the code.
        """
        if available_dataframes:
            self.intelligent_fixer.available_dataframes = available_dataframes
        
        # Get intelligent suggestions
        intelligent_suggestions = self.intelligent_fixer.generate_intelligent_suggestions(code)
        
        # Add DataFrame-specific suggestions
        df_suggestions = []
        if available_dataframes:
            df_names = list(available_dataframes.keys())
            df_suggestions.append(f"Available DataFrames: {', '.join(df_names)}")
            
            # Show sample data from first DataFrame
            if df_names:
                first_df = available_dataframes[df_names[0]]
                if hasattr(first_df, 'columns'):
                    sample_cols = list(first_df.columns)[:5]
                    df_suggestions.append(f"Sample columns in {df_names[0]}: {', '.join(sample_cols)}")
        
        return intelligent_suggestions + df_suggestions
    
    def validate_and_test_code(self, code: str, available_dataframes: Dict[str, Any] = None) -> Tuple[bool, List[str], str]:
        """
        Validate the corrected code and test if it can be executed safely.
        
        Returns:
            Tuple of (is_valid, warnings, error_message)
        """
        warnings = []
        error_message = ""
        
        try:
            # Basic syntax check
            compile(code, '<string>', 'exec')
            
            # Check for common issues that might cause runtime errors
            if 'idxmax()' in code and 'groupby' in code:
                # Check if groupby result is properly handled
                if '.idxmax()' in code and not any(pattern in code for pattern in ['.reset_index()', '.loc[']):
                    warnings.append("Consider using .reset_index() after groupby operations for better results")
            
            if 'loc[' in code and 'idxmax()' in code:
                # Check if loc access is properly formatted
                if re.search(r'\.loc\[[^\]]*idxmax\(\)[^\]]*\]', code):
                    warnings.append("Make sure to use .loc[] properly with idxmax() results")
            
            return True, warnings, error_message
            
        except SyntaxError as e:
            error_message = f"Syntax error: {str(e)}"
            return False, warnings, error_message
        except Exception as e:
            error_message = f"Code validation error: {str(e)}"
            return False, warnings, error_message
    
    def fix_code_with_context(self, code: str, user_query: str, available_dataframes: Dict[str, Any] = None) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Fix code with full context including user query and available DataFrames.
        """
        if available_dataframes:
            self.intelligent_fixer.available_dataframes = available_dataframes
        
        # Analyze the full context
        context = self.intelligent_fixer.analyze_code_context(code)
        context['user_query'] = user_query
        context['available_dataframes'] = list(available_dataframes.keys()) if available_dataframes else []
        
        # Apply intelligent fixes
        fixed_code, fixes = self.intelligent_fixer.fix_code_dynamically(code, available_dataframes)
        
        # Apply additional context-aware fixes
        context_fixed_code, context_fixes = self._apply_context_aware_fixes(fixed_code, context, available_dataframes)
        fixes.extend(context_fixes)
        
        # Validate the final code
        is_valid, warnings, error_message = self.validate_and_test_code(context_fixed_code, available_dataframes)
        
        if not is_valid:
            # If validation fails, try to fix syntax errors
            syntax_fixed_code, syntax_fixes = self._fix_syntax_errors(context_fixed_code)
            fixes.extend(syntax_fixes)
            
            # Re-validate after syntax fixes
            is_valid, warnings, error_message = self.validate_and_test_code(syntax_fixed_code, available_dataframes)
            
            if is_valid:
                context_fixed_code = syntax_fixed_code
            else:
                # Add the error message to context for debugging
                context['validation_error'] = error_message
                context['validation_warnings'] = warnings
        
        return context_fixed_code, fixes, context
    
    def _apply_context_aware_fixes(self, code: str, context: Dict[str, Any], available_dataframes: Dict[str, Any] = None) -> Tuple[str, List[str]]:
        """
        Apply fixes based on the full context including user query.
        """
        fixes = []
        
        # If user is asking about battery capacity, ensure we're using the right columns
        if context.get('user_query', '').lower().find('battery') != -1:
            if available_dataframes:
                for df_name, df in available_dataframes.items():
                    if hasattr(df, 'columns'):
                        battery_cols = [col for col in df.columns if 'battery' in col.lower()]
                        if battery_cols:
                            # Ensure we're using battery columns
                            if not any(col in code.lower() for col in battery_cols):
                                fixes.append(f"Consider using battery-related columns: {battery_cols[:3]}")
        
        # If user is asking about maximum/minimum, ensure proper pattern
        if context.get('intent') == 'find_maximum':
            if 'idxmax' in code and 'loc' not in code:
                # Suggest proper maximum finding pattern
                fixes.append("Consider using .loc[] to access the row with maximum value")
        
        return code, fixes 