#!/usr/bin/env python3
"""
Quick runner script for all visualizations
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_comparison_analysis import main

if __name__ == "__main__":
    print("🎨 Running Satellite Image Super-Resolution Visualization System")
    print("=" * 60)
    main()
