"""
Test package for utils modules.
"""

import sys
import os

# Add the project root to the path so tests can import utils modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import test modules
from . import test_simple_fix
from . import test_fixed_dataframe_correction
from . import test_dataframe_fixes
from . import test_dataframe_fixes_final
from . import test_dynamic_dataframe_fixes
from . import test_intelligent_fixing
from . import test_performance_integration
from . import test_diagnostics
from . import test_auth_url
from . import test_oauth_flow
from . import test_graph_permissions
from . import test_auth_fixes
from . import test_azure_auth
from . import test_qdrant_error_capture
from . import test_qdrant_app_simulation
from . import test_qdrant_debug
from . import test_qdrant_app_issue
from . import test_qdrant_connection
from . import test_date_filter_handling
from . import test_pyodbc_conn
from . import test_syntax_error_correction
from . import test_mongodb_integration

__all__ = [
    'test_simple_fix',
    'test_fixed_dataframe_correction',
    'test_dataframe_fixes',
    'test_dataframe_fixes_final',
    'test_dynamic_dataframe_fixes',
    'test_intelligent_fixing',
    'test_performance_integration',
    'test_diagnostics',
    'test_auth_url',
    'test_oauth_flow',
    'test_graph_permissions',
    'test_auth_fixes',
    'test_azure_auth',
    'test_qdrant_error_capture',
    'test_qdrant_app_simulation',
    'test_qdrant_debug',
    'test_qdrant_app_issue',
    'test_qdrant_connection',
    'test_date_filter_handling',
    'test_pyodbc_conn',
    'test_syntax_error_correction',
    'test_mongodb_integration',
] 