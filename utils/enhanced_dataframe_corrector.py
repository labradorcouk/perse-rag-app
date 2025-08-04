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
    
    def correct_dataframe_names(self, code: str, available_dataframes: Dict[str, Any] = None) -> Tuple[str, List[str]]:
        """
        Intelligently correct DataFrame names using both pattern matching and intelligent analysis.
        """
        if available_dataframes:
            self.intelligent_fixer.available_dataframes = available_dataframes
        
        # First, apply traditional pattern-based corrections for DataFrame references only
        corrected_code, pattern_fixes = self._apply_pattern_corrections(code, available_dataframes)
        
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
        all_fixes = pattern_fixes + intelligent_fixes
        
        return corrected_code, all_fixes
    
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