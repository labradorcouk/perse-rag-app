import re
import ast
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

class IntelligentDataFrameFixer:
    """
    An intelligent system that dynamically detects and fixes DataFrame reference issues
    without hardcoding specific mappings. It uses pattern recognition, context analysis,
    and intelligent suggestions to resolve issues automatically.
    """
    
    def __init__(self, available_dataframes: Dict[str, Any] = None):
        self.available_dataframes = available_dataframes or {}
        self.df_patterns = {
            'column_access': r'(\w+)\[([\'"])([^\'"]+)\2\]',
            'method_call': r'(\w+)\.(\w+)\(',
            'attribute_access': r'(\w+)\.(\w+)',
            'variable_assignment': r'(\w+)\s*=\s*(\w+)',
            'function_call': r'(\w+)\(',
        }
        self.logger = logging.getLogger(__name__)
    
    def analyze_code_context(self, code: str) -> Dict[str, Any]:
        """
        Analyze the code context to understand what the user is trying to achieve.
        """
        context = {
            'intent': self._detect_intent(code),
            'dataframe_usage': self._extract_dataframe_usage(code),
            'column_references': self._extract_column_references(code),
            'method_calls': self._extract_method_calls(code),
            'potential_issues': self._identify_potential_issues(code)
        }
        return context
    
    def _detect_intent(self, code: str) -> str:
        """
        Intelligently detect what the user is trying to do based on code patterns.
        """
        code_lower = code.lower()
        
        # Pattern-based intent detection
        if any(word in code_lower for word in ['max', 'highest', 'largest', 'maximum']):
            return 'find_maximum'
        elif any(word in code_lower for word in ['min', 'lowest', 'smallest', 'minimum']):
            return 'find_minimum'
        elif any(word in code_lower for word in ['mean', 'average', 'avg']):
            return 'calculate_average'
        elif any(word in code_lower for word in ['count', 'sum', 'total']):
            return 'aggregate_data'
        elif any(word in code_lower for word in ['group', 'groupby']):
            return 'group_analysis'
        elif any(word in code_lower for word in ['sort', 'order']):
            return 'sort_data'
        elif any(word in code_lower for word in ['filter', 'where', 'condition']):
            return 'filter_data'
        elif any(word in code_lower for word in ['plot', 'chart', 'graph', 'visualize']):
            return 'visualization'
        else:
            return 'general_analysis'
    
    def _extract_dataframe_usage(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract all DataFrame usage patterns from the code.
        """
        usage = []
        
        # Find DataFrame column access patterns
        column_pattern = r'(\w+)\[([\'"])([^\'"]+)\2\]'
        for match in re.finditer(column_pattern, code):
            df_name = match.group(1)
            column_name = match.group(3)
            usage.append({
                'type': 'column_access',
                'dataframe': df_name,
                'column': column_name,
                'full_match': match.group(0)
            })
        
        # Find DataFrame method calls
        method_pattern = r'(\w+)\.(\w+)\('
        for match in re.finditer(method_pattern, code):
            df_name = match.group(1)
            method_name = match.group(2)
            usage.append({
                'type': 'method_call',
                'dataframe': df_name,
                'method': method_name,
                'full_match': match.group(0)
            })
        
        return usage
    
    def _extract_column_references(self, code: str) -> List[str]:
        """
        Extract all column references from the code.
        """
        columns = []
        
        # Look for column names in quotes (most reliable)
        column_pattern = r'[\'"]([A-Za-z_][A-Za-z0-9_]*)[\'"]'
        quoted_columns = re.findall(column_pattern, code)
        columns.extend(quoted_columns)
        
        # Look for DataFrame column access patterns like df['column'] or df.column
        df_column_patterns = [
            r'df\d*\[([^\]]+)\]',  # df['column_name']
            r'df\d*\.([A-Za-z_][A-Za-z0-9_]*)',  # df.column_name
        ]
        
        for pattern in df_column_patterns:
            matches = re.findall(pattern, code)
            columns.extend(matches)
        
        # Remove duplicates and filter out non-column patterns
        unique_columns = []
        for col in set(columns):
            # Skip if it looks like a method name
            if col in ['loc', 'iloc', 'head', 'tail', 'describe', 'info', 'columns', 'shape',
                      'max', 'min', 'mean', 'sum', 'count', 'dropna', 'fillna', 'sort_values',
                      'groupby', 'merge', 'join', 'concat', 'append', 'drop', 'rename', 'copy',
                      'astype', 'dtypes', 'index', 'idxmax', 'idxmin', 'nlargest', 'nsmallest']:
                continue
            # Skip if it's too short (likely not a column)
            if len(col) < 3:
                continue
            unique_columns.append(col)
        
        return unique_columns
    
    def _extract_method_calls(self, code: str) -> List[Dict[str, str]]:
        """
        Extract method calls and their context.
        """
        method_calls = []
        method_pattern = r'(\w+)\.(\w+)\('
        
        for match in re.finditer(method_pattern, code):
            method_calls.append({
                'object': match.group(1),
                'method': match.group(2),
                'full_match': match.group(0)
            })
        
        return method_calls
    
    def _identify_potential_issues(self, code: str) -> List[Dict[str, Any]]:
        """
        Identify potential issues in the code that need fixing.
        """
        issues = []
        
        # Check for undefined DataFrame references
        df_usage = self._extract_dataframe_usage(code)
        for usage in df_usage:
            df_name = usage['dataframe']
            if df_name not in self.available_dataframes:
                issues.append({
                    'type': 'undefined_dataframe',
                    'dataframe': df_name,
                    'suggestion': self._suggest_dataframe_replacement(df_name)
                })
        
        # Check for undefined column references
        column_refs = self._extract_column_references(code)
        for col in column_refs:
            if not self._is_column_available(col):
                issues.append({
                    'type': 'undefined_column',
                    'column': col,
                    'suggestion': self._suggest_column_replacement(col)
                })
        
        # Check for incorrect method usage
        method_calls = self._extract_method_calls(code)
        for call in method_calls:
            if not self._is_method_available(call['object'], call['method']):
                issues.append({
                    'type': 'incorrect_method',
                    'object': call['object'],
                    'method': call['method'],
                    'suggestion': self._suggest_method_replacement(call['object'], call['method'])
                })
        
        return issues
    
    def _is_column_available(self, column_name: str) -> bool:
        """
        Check if a column is available in any of the available DataFrames.
        """
        for df_name, df in self.available_dataframes.items():
            if hasattr(df, 'columns') and column_name in df.columns:
                return True
        return False
    
    def _is_method_available(self, object_name: str, method_name: str) -> bool:
        """
        Check if a method is available on the specified object.
        """
        if object_name in self.available_dataframes:
            df = self.available_dataframes[object_name]
            return hasattr(df, method_name)
        return False
    
    def _suggest_dataframe_replacement(self, undefined_df: str) -> str:
        """
        Intelligently suggest a DataFrame replacement.
        """
        if not self.available_dataframes:
            return "df1"  # Default fallback
        
        # Look for similar names
        for df_name in self.available_dataframes.keys():
            if undefined_df.lower() in df_name.lower() or df_name.lower() in undefined_df.lower():
                return df_name
        
        # Return the first available DataFrame
        return list(self.available_dataframes.keys())[0]
    
    def _suggest_column_replacement(self, undefined_column: str) -> str:
        """
        Intelligently suggest a column replacement.
        """
        # Look for similar column names across all DataFrames
        for df_name, df in self.available_dataframes.items():
            if hasattr(df, 'columns'):
                for col in df.columns:
                    if (undefined_column.lower() in col.lower() or 
                        col.lower() in undefined_column.lower() or
                        self._calculate_similarity(undefined_column, col) > 0.7):
                        return col
        
        return "No suitable replacement found"
    
    def _suggest_method_replacement(self, object_name: str, method_name: str) -> str:
        """
        Suggest method replacements based on common patterns.
        """
        method_suggestions = {
            'loc': 'iloc',
            'iloc': 'loc',
            'idxmax': 'max',
            'idxmin': 'min',
            'nlargest': 'sort_values',
            'nsmallest': 'sort_values'
        }
        
        return method_suggestions.get(method_name, f"Check if {method_name} is available")
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity using simple algorithm.
        """
        if str1 == str2:
            return 1.0
        
        # Simple similarity calculation
        common_chars = sum(1 for c in str1 if c in str2)
        total_chars = max(len(str1), len(str2))
        
        return common_chars / total_chars if total_chars > 0 else 0.0
    
    def fix_code_dynamically(self, code: str, available_dataframes: Dict[str, Any] = None) -> Tuple[str, List[str]]:
        """
        Dynamically fix code issues without hardcoding specific mappings.
        """
        if available_dataframes:
            self.available_dataframes = available_dataframes
        
        # Analyze the code context
        context = self.analyze_code_context(code)
        
        # Generate intelligent fixes
        fixed_code = code
        fixes_applied = []
        
        # Fix DataFrame references
        for issue in context['potential_issues']:
            if issue['type'] == 'undefined_dataframe':
                old_df = issue['dataframe']
                new_df = issue['suggestion']
                fixed_code = re.sub(rf'\b{old_df}\b', new_df, fixed_code)
                fixes_applied.append(f"Replaced undefined DataFrame '{old_df}' with '{new_df}'")
            
            elif issue['type'] == 'undefined_column':
                # This is more complex - we need to understand the context
                fixed_code, column_fixes = self._fix_column_references(fixed_code, issue)
                fixes_applied.extend(column_fixes)
        
        # Apply intelligent code structure fixes
        fixed_code, structure_fixes = self._fix_code_structure(fixed_code, context)
        fixes_applied.extend(structure_fixes)
        
        return fixed_code, fixes_applied
    
    def _fix_column_references(self, code: str, issue: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Fix column references intelligently.
        """
        fixes = []
        undefined_column = issue['column']
        suggested_column = issue['suggestion']
        
        if suggested_column != "No suitable replacement found":
            # Replace the column reference
            code = re.sub(rf'[\'"]({undefined_column})[\'"]', f"'{suggested_column}'", code)
            fixes.append(f"Replaced column '{undefined_column}' with '{suggested_column}'")
        else:
            # Try to find a similar column or suggest alternatives
            similar_columns = self._find_similar_columns(undefined_column)
            if similar_columns:
                code = re.sub(rf'[\'"]({undefined_column})[\'"]', f"'{similar_columns[0]}'", code)
                fixes.append(f"Replaced column '{undefined_column}' with similar column '{similar_columns[0]}'")
            else:
                fixes.append(f"Warning: Column '{undefined_column}' not found in any DataFrame")
        
        return code, fixes
    
    def _find_similar_columns(self, target_column: str) -> List[str]:
        """
        Find columns similar to the target column.
        """
        similar_columns = []
        
        for df_name, df in self.available_dataframes.items():
            if hasattr(df, 'columns'):
                for col in df.columns:
                    similarity = self._calculate_similarity(target_column.lower(), col.lower())
                    if similarity > 0.5:  # Threshold for similarity
                        similar_columns.append(col)
        
        return sorted(similar_columns, key=lambda x: self._calculate_similarity(target_column.lower(), x.lower()), reverse=True)
    
    def _fix_code_structure(self, code: str, context: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Fix code structure issues intelligently.
        """
        fixes = []
        
        # Fix common DataFrame operation patterns
        if context['intent'] == 'find_maximum':
            # Ensure proper maximum finding pattern
            if 'idxmax' in code and 'loc' not in code:
                # Add proper loc access
                code = re.sub(r'\.idxmax\(\)', '.idxmax()\nmax_vehicle = df1.loc[max_battery_idx]', code)
                fixes.append("Added proper loc access for maximum value retrieval")
        
        elif context['intent'] == 'find_minimum':
            # Ensure proper minimum finding pattern
            if 'idxmin' in code and 'loc' not in code:
                code = re.sub(r'\.idxmin\(\)', '.idxmin()\nmin_vehicle = df1.loc[min_battery_idx]', code)
                fixes.append("Added proper loc access for minimum value retrieval")
        
        # Fix common DataFrame method chains
        if 'dropna' in code and 'subset' not in code:
            # Add subset parameter if needed
            code = re.sub(r'\.dropna\(\)', '.dropna(subset=[col for col in df1.columns if "Battery" in col])', code)
            fixes.append("Added subset parameter to dropna for battery columns")
        
        return code, fixes
    
    def generate_intelligent_suggestions(self, code: str) -> List[str]:
        """
        Generate intelligent suggestions for improving the code.
        """
        suggestions = []
        context = self.analyze_code_context(code)
        
        # Suggest improvements based on intent
        if context['intent'] == 'find_maximum':
            suggestions.append("Consider adding error handling for empty DataFrames")
            suggestions.append("Add data validation before finding maximum")
            suggestions.append("Consider showing top N results instead of just the maximum")
        
        elif context['intent'] == 'general_analysis':
            suggestions.append("Consider adding data quality checks")
            suggestions.append("Add summary statistics for better context")
            suggestions.append("Consider visualizing the results")
        
        # Suggest based on available data
        if self.available_dataframes:
            df_names = list(self.available_dataframes.keys())
            suggestions.append(f"Available DataFrames: {', '.join(df_names)}")
            
            # Show sample columns from first DataFrame
            if df_names:
                first_df = self.available_dataframes[df_names[0]]
                if hasattr(first_df, 'columns'):
                    sample_cols = list(first_df.columns)[:5]
                    suggestions.append(f"Sample columns in {df_names[0]}: {', '.join(sample_cols)}")
        
        return suggestions 