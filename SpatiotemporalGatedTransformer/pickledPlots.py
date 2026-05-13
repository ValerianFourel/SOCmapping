import pickle
import sys
from pathlib import Path

def inspect_pickle(filepath):
    """
    Inspect and display contents of a pickle file
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Error: File not found at {filepath}")
        return
    
    print(f"Inspecting pickle file: {filepath}")
    print(f"File size: {filepath.stat().st_size / 1024:.2f} KB")
    print("-" * 80)
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nData type: {type(data)}")
        print(f"Data structure: {type(data).__name__}")
        
        # Handle different data types
        if isinstance(data, dict):
            print(f"\nDictionary with {len(data)} keys:")
            for key in data.keys():
                value = data[key]
                print(f"\n  Key: '{key}'")
                print(f"    Type: {type(value)}")
                
                # Show shape/length for common types
                if hasattr(value, 'shape'):
                    print(f"    Shape: {value.shape}")
                elif hasattr(value, '__len__') and not isinstance(value, str):
                    print(f"    Length: {len(value)}")
                
                # Show preview of content
                if isinstance(value, (list, tuple)) and len(value) <= 10:
                    print(f"    Content: {value}")
                elif isinstance(value, (int, float, str, bool)):
                    print(f"    Value: {value}")
                elif hasattr(value, 'shape'):  # numpy array or similar
                    print(f"    Preview: {value}")
                    
        elif isinstance(data, (list, tuple)):
            print(f"\n{type(data).__name__} with {len(data)} elements")
            for i, item in enumerate(data[:5]):  # Show first 5 items
                print(f"\n  Element {i}:")
                print(f"    Type: {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"    Shape: {item.shape}")
                    
        else:
            print(f"\nContent preview:")
            print(data)
            
    except Exception as e:
        print(f"\nError loading pickle file: {e}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    filepath = "/home/valerian/SGTPublication/residual_Maps_Bavaria_360kTFT/residual_Maps_Bavaria_360kTFT/analysis_results.pkl"
    inspect_pickle(filepath)
