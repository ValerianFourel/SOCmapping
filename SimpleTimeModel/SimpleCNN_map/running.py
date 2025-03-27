import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os

from dataloader.dataloader import MultiRasterDataset 
from dataloader.dataloaderMapping import MultiRasterDatasetMapping  # Updated class above
from dataloader.dataframe_loader import filter_dataframe, separate_and_add_data
from config import (TIME_BEGINNING, TIME_END, INFERENCE_TIME, seasons, years_padded, 
                   SamplesCoordinates_Yearly, MatrixCoordinates_1mil_Yearly, 
                   DataYearly, SamplesCoordinates_Seasonally, 
                   MatrixCoordinates_1mil_Seasonally, DataSeasonally, 
                   file_path_LUCAS_LFU_Lfl_00to23_Bavaria_OC, 
                   window_size, bands_list_order)
from mapping import create_prediction_visualizations
from modelCNN import SmallCNN


def load_cnn_model(model_path="/home/vfourel/SOCProject/SOCmapping/SimpleTimeModel/SimpleCNN_map/simpletimecnn_model_MAX_OC_150_TIME_BEGINNING_2007_TIME_END_2023.pth"):
    print("1. Entering load_cnn_model")
    try:
        model = SmallCNN()
        print("2. Model initialized:", model.__class__.__name__)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        print("3. State dict loaded with keys:", list(state_dict.keys())[:5])
        
        if all(k.startswith('module.') for k in state_dict.keys()):
            print("4. Removing 'module.' prefix from state_dict keys")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif not any(k.startswith('module.') for k in state_dict.keys()) and hasattr(model, 'module'):
            print("5. Adding 'module.' prefix to state_dict keys")
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        print("6. State dict loaded into model")
        model.eval()
        print("7. Model set to evaluation mode")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def separate_and_add_data_1mil_inference(TIME_END=INFERENCE_TIME, 
                                        SamplesCoordinates_Yearly=MatrixCoordinates_1mil_Yearly):
    print("8. Entering separate_and_add_data_1mil_inference")
    
    def create_path_arrays_yearly(paths, year):
        samples_paths = []
        for path in paths:
            if 'Elevation' in path:
                samples_paths.append(path)  # Elevation is static
            else:
                samples_paths.append(f"{path}/{year}")
        return samples_paths

    samples_paths = create_path_arrays_yearly(SamplesCoordinates_Yearly, TIME_END)
    
    print("9. Returning samples paths:", len(samples_paths))
    return samples_paths


def flatten_paths(path_list):
    print("10. Entering flatten_paths with input length:", len(path_list))
    flattened = [item for sublist in path_list for item in (flatten_paths(sublist) if isinstance(sublist, list) else [sublist])]
    print("11. Flattened paths length:", len(flattened))
    return flattened


def accelerated_predict(df_full, cnn_model, subfolders, batch_size=8):
    print("12. Entering accelerated_predict")
    print("13. df_full shape:", df_full.shape)
    print("14. subfolders:", subfolders[:5])
    
    try:
        accelerator = Accelerator()
        print("15. Accelerator initialized")

        print("16. Creating MultiRasterDatasetMapping")
        dataset = MultiRasterDatasetMapping(subfolders, df_full, bands_list_order=bands_list_order)
        print("17. Dataset length:", len(dataset))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        print("18. DataLoader created with batch_size:", batch_size)

        model, dataloader = accelerator.prepare(cnn_model, dataloader)
        print("19. Model and DataLoader prepared with accelerator")

        all_predictions = []
        all_coordinates = []

        with torch.no_grad():
            print("20. Starting inference loop")
            for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                inputs, coords = batch  # inputs: (batch, num_bands, h, w), coords: (batch, 2)


                outputs = model(inputs)

                gathered_outputs = accelerator.gather(outputs)
                gathered_coords = accelerator.gather(coords)

                if accelerator.is_main_process:
                    all_predictions.append(gathered_outputs.cpu().numpy())
                    all_coordinates.append(gathered_coords.cpu().numpy())

        if accelerator.is_main_process:
            predictions = np.vstack(all_predictions)
            coordinates = np.vstack(all_coordinates)
            print("27. Final predictions shape:", predictions.shape)
            print("28. Final coordinates shape:", coordinates.shape)
            return coordinates, predictions
        else:
            print("29. Not main process, returning None")
            return None, None
            
    except Exception as e:
        print(f"Error in accelerated prediction: {e}")
        return None, None
    finally:
        if 'accelerator' in locals():
            accelerator.free_memory()
            accelerator.wait_for_everyone()


def main():
    print("30. Entering main")
    try:
        subfolders = separate_and_add_data_1mil_inference()
        print("31. Subfolders retrieved:", len(subfolders))
        
        subfolders = flatten_paths(subfolders)
        print("32. Paths flattened:", len(subfolders))

        subfolders = list(dict.fromkeys(subfolders))
        print("33. Duplicates removed:", len(subfolders))

        file_path = "/home/vfourel/SOCProject/SOCmapping/Data/Coordinates1Mil/coordinates_Bavaria_1mil.csv"
        df_full = pd.read_csv(file_path)
        print("34. Coordinates loaded, head:\n", df_full.head())

        cnn_model = load_cnn_model()
        print("35. CNN model loaded")

        coordinates, predictions = accelerated_predict(
            df_full=df_full,
            cnn_model=cnn_model,
            subfolders=subfolders,
            batch_size=512
        )
        print("36. Prediction completed")

        if coordinates is not None and predictions is not None:
            save_path_coords = "coordinates_1mil.npy"
            save_path_preds = "predictions_1mil.npy"

            np.save(save_path_coords, coordinates)
            np.save(save_path_preds, predictions)
            print("37. Results saved to:", save_path_coords, save_path_preds)

            save_path = '/home/vfourel/SOCProject/SOCmapping/predictions_plots/cnnsimple_plots'
            Path(save_path).mkdir(parents=True, exist_ok=True)
            print("38. Visualization directory created:", save_path)
            create_prediction_visualizations(INFERENCE_TIME, coordinates, predictions, save_path)
            print(f"39. Results saved and visualizations created at {save_path}")
        else:
            print("40. Prediction failed or not on main process")

    except Exception as e:
        print(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()