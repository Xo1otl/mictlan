import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from funsearch import function
from funsearch.datadriven import Dataset, enhanced_dataset_evaluator


@dataclass
class SkeletonInfo:
    index: int
    skeleton: function.Skeleton
    optimal_params: np.ndarray
    score: float
    mse: float
    description: str


class OneDimensionalPlotComponent:
    def __init__(self, dataset: Dataset):
        if dataset.inputs.shape[1] != 1:
            raise ValueError(
                f"OneDimensionalPlotComponent only supports 1D input data, got {dataset.inputs.shape[1]}D")
        self.dataset = dataset
        self.skeletons: Dict[int, SkeletonInfo] = {}
        self.selected_skeletons: List[int] = []

    def add_skeleton(self, index: int, skeleton: function.Skeleton, description: str = "") -> bool:
        try:
            result = enhanced_dataset_evaluator(skeleton, self.dataset)
            self.skeletons[index] = SkeletonInfo(
                index=index,
                skeleton=skeleton,
                optimal_params=result.optimal_params,
                score=result.score,
                mse=result.mse,
                description=description or f"Function {index}"
            )
            return True
        except Exception as e:
            print(f"Error adding skeleton {index}: {e}")
            return False

    def get_available_skeletons(self) -> List[Tuple[int, str, float]]:
        return [(info.index, info.description, info.score)
                for info in self.skeletons.values()]

    def select_skeletons(self, indices: List[int]) -> None:
        self.selected_skeletons = [i for i in indices if i in self.skeletons]

    def generate_predictions(self, skeleton_info: SkeletonInfo,
                             params: Optional[np.ndarray] = None,
                             x_input: Optional[np.ndarray] = None) -> np.ndarray:
        if params is None:
            params = skeleton_info.optimal_params

        if x_input is None:
            x_input = self.dataset.inputs.flatten()

        try:
            return skeleton_info.skeleton(x_input, params)
        except Exception as e:
            print(f"Error generating predictions: {e}")
            return np.array([])

    def create_plot_data(self, param_adjustments: Optional[Dict[int, np.ndarray]] = None) -> Dict[str, Any]:
        if param_adjustments is None:
            param_adjustments = {}

        # Generate high-resolution x values for smooth function curves
        x_data_original = self.dataset.inputs.flatten()
        x_min, x_max = x_data_original.min(), x_data_original.max()
        x_range = x_max - x_min
        x_smooth = np.linspace(x_min - 0.1 * x_range,
                               x_max + 0.1 * x_range, 1000)

        plot_data = {
            'x_data': x_data_original,
            'y_actual': self.dataset.outputs,
            'x_smooth': x_smooth,
            'functions': []
        }

        for idx in self.selected_skeletons:
            skeleton_info = self.skeletons[idx]
            params = param_adjustments.get(idx, skeleton_info.optimal_params)

            # Generate predictions on original data points for MSE calculation
            y_pred_original = self.generate_predictions(
                skeleton_info, params, x_data_original)
            # Generate smooth curve for visualization
            y_pred_smooth = self.generate_predictions(
                skeleton_info, params, x_smooth)

            if len(y_pred_original) > 0 and len(y_pred_smooth) > 0:
                plot_data['functions'].append({
                    'index': idx,
                    'description': skeleton_info.description,
                    'y_pred_smooth': y_pred_smooth,
                    'params': params,
                    'mse': skeleton_info.mse,
                    'original_mse': skeleton_info.mse
                })

        return plot_data

    def get_param_bounds(self, skeleton_idx: int) -> List[Tuple[float, float]]:
        if skeleton_idx not in self.skeletons:
            return []

        params = self.skeletons[skeleton_idx].optimal_params
        return [(param - abs(param) * 2.0 if param != 0 else -10.0,
                 param + abs(param) * 2.0 if param != 0 else 10.0)
                for param in params]


def create_matplotlib_plot(plot_data: Dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    x_data = plot_data['x_data']
    y_actual = plot_data['y_actual']
    x_smooth = plot_data['x_smooth']

    y_label = 'Output'
    # Check if y_actual contains complex numbers
    if np.iscomplexobj(y_actual):
        y_actual = np.abs(y_actual)
        y_label = 'Absolute Value of Output'

    ax.scatter(x_data, y_actual, color='black', s=50,
               alpha=0.7, label='Actual Data', zorder=5)

    colors = ['red', 'blue', 'green', 'orange',
              'purple', 'brown', 'pink', 'gray']

    for i, func_data in enumerate(plot_data['functions']):
        color = colors[i % len(colors)]
        mse_text = f"MSE: {func_data['mse']:.4f}"
        label = f"{func_data['description']} ({mse_text})"

        y_pred_smooth = func_data['y_pred_smooth']
        if np.iscomplexobj(y_pred_smooth):
            y_pred_smooth = np.abs(y_pred_smooth)
            y_label = 'Absolute Value of Output'

        ax.plot(x_smooth, y_pred_smooth, color=color, linewidth=2,
                label=label, alpha=0.8)

    ax.set_xlabel('Input')
    ax.set_ylabel(y_label)
    ax.set_title('Function Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
