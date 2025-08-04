#!/usr/bin/env python3
"""
Dynamic DataFrame Name Correction Utility

This module provides intelligent DataFrame name correction that automatically
detects and fixes DataFrame name issues without hardcoding table names.
"""

import re
import pandas as pd
from typing import Dict, List, Any, Tuple
from .enhanced_dataframe_corrector import EnhancedDataFrameCorrector

class DataFrameCorrector:
    """
    Enhanced DataFrame corrector that combines intelligent fixing with existing correction logic.
    This system can dynamically handle any DataFrame reference issues without hardcoding specific mappings.
    """
    
    def __init__(self):
        self.enhanced_corrector = EnhancedDataFrameCorrector()
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
        return self.enhanced_corrector.correct_dataframe_names(code, available_dataframes)
    
    def validate_dataframe_references(self, code: str, available_dataframes: Dict[str, Any] = None) -> List[str]:
        """
        Validate DataFrame references and return any unresolved issues.
        """
        # Only check for actual DataFrame references, not column names
        issues = []
        
        if not available_dataframes:
            return ["No DataFrames available"]
        
        # Look for DataFrame usage patterns
        df_patterns = [
            r'\b(\w+)\[',  # df[
            r'\b(\w+)\.(head|tail|describe|info|columns|shape|loc|iloc|max|min|mean|sum|count|dropna|fillna|sort_values|groupby|merge|join|concat|append|drop|rename|copy|astype|dtypes|index|idxmax|idxmin|nlargest|nsmallest)\b'
        ]
        
        for pattern in df_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                if isinstance(match, tuple):
                    df_name = match[0]
                else:
                    df_name = match
                
                # Skip if it's a known DataFrame or common variable
                if df_name in available_dataframes or df_name in ['pd', 'np', 'plt', 'result', 'fig']:
                    continue
                
                # Check if it looks like a DataFrame reference
                if self._looks_like_dataframe_reference(df_name, code):
                    if df_name not in available_dataframes:
                        issues.append(f"Undefined DataFrame: {df_name}")
        
        return issues
    
    def get_dataframe_info(self, available_dataframes: Dict[str, Any] = None) -> str:
        """
        Get comprehensive information about available DataFrames for LLM context.
        """
        return self.enhanced_corrector.get_dataframe_info(available_dataframes)
    
    def generate_smart_suggestions(self, code: str, available_dataframes: Dict[str, Any] = None) -> List[str]:
        """
        Generate smart suggestions for improving the code.
        """
        return self.enhanced_corrector.generate_smart_suggestions(code, available_dataframes)
    
    def fix_code_with_context(self, code: str, user_query: str, available_dataframes: Dict[str, Any] = None) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Fix code with full context including user query and available DataFrames.
        This is the main method that should be used for intelligent code fixing.
        """
        return self.enhanced_corrector.fix_code_with_context(code, user_query, available_dataframes)
    
    # Legacy methods for backward compatibility
    def correct_dataframe_names_legacy(self, code: str, available_dataframes: Dict[str, Any] = None) -> Tuple[str, List[str]]:
        """
        Legacy method for backward compatibility.
        """
        if available_dataframes is None:
            available_dataframes = {}
        
        # Find potential DataFrame references
        potential_dfs = self._find_potential_dataframe_references(code)
        
        # Get preferred DataFrame name
        preferred_df = self._get_preferred_dataframe(available_dataframes)
        
        # Replace potential DataFrame references
        fixes = []
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