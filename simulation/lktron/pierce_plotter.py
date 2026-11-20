"""
Pierce Data Plotter - COMPLETE VERSION
--------------------
Features:
- Multiple plots separated by color_id 0
- Does not plot color_id 0 (reference only)
- Proper legends for indices 1-6
- Z and R axes for pierce gun plots
- Specific axes for current density and space charge plots
- 9:16 aspect ratio with uniform scaling
- Footer with run number and timestamp
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DataSegment:
    """Represents a single line/curve segment."""
    color_id: int
    num_points: int
    x_coords: List[float]
    y_coords: List[float]


class PierceDataParser:
    """Parses pierce.txt format files."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.segments: List[DataSegment] = []
    
    def parse(self) -> List[DataSegment]:
        """Parse the file and extract all data segments."""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            parts = line.split()
            if len(parts) == 4:
                try:
                    color_id = int(parts[0])
                    num_points = int(parts[1])
                    header_x = float(parts[2])
                    header_y = float(parts[3])
                    
                    x_coords = []
                    y_coords = []
                    
                    if num_points == 1:
                        x_coords.append(header_x)
                        y_coords.append(header_y)
                        i += 1
                    else:
                        i += 1
                        coords_collected = []
                        
                        while len(x_coords) < num_points and i < len(lines):
                            coord_line = lines[i].strip()
                            if not coord_line:
                                i += 1
                                continue
                            
                            test_parts = coord_line.split()
                            if len(test_parts) == 4:
                                try:
                                    test_id = int(test_parts[0])
                                    test_num = int(test_parts[1])
                                    break
                                except ValueError:
                                    pass
                            
                            values = coord_line.split()
                            for val in values:
                                try:
                                    coords_collected.append(float(val))
                                except ValueError:
                                    pass
                            
                            i += 1
                        
                        for j in range(0, len(coords_collected), 2):
                            if j + 1 < len(coords_collected):
                                x_coords.append(coords_collected[j])
                                y_coords.append(coords_collected[j + 1])
                                if len(x_coords) >= num_points:
                                    break
                    
                    segment = DataSegment(
                        color_id=color_id,
                        num_points=num_points,
                        x_coords=x_coords,
                        y_coords=y_coords
                    )
                    self.segments.append(segment)
                    
                except (ValueError, IndexError):
                    i += 1
            else:
                i += 1
        
        return self.segments


class PiercePlotter:
    """Plots parsed pierce data with multiple plots separated by color_id 0."""
    
    def __init__(self, segments: List[DataSegment], run_no: str = "001"):
        self.segments = segments
        self.run_no = run_no
        self.color_map = {
            1: '#000000',  # black - boundary
            2: '#d62728',  # red - electron trajectory
            3: '#2ca02c',  # green - equipotential
            4: '#1f77b4',  # blue - magnetic field
            5: '#9467bd',  # purple - current density
            6: '#ff7f0e',  # orange - space charge
        }
        self.legend_map = {
            1: 'Boundary',
            2: 'Electron Trajectory',
            3: 'Equipotential Line',
            4: 'Magnetic Field Profile',
            5: 'Current Density Distribution',
            6: 'Space Charge Density Distribution'
        }
    
    def _split_into_plots(self) -> List[List[DataSegment]]:
        """Split segments into separate plots based on color_id 0 markers."""
        plots = []
        current_plot = []
        
        for segment in self.segments:
            if segment.color_id == 0:
                if current_plot:
                    plots.append(current_plot)
                current_plot = []
            else:
                current_plot.append(segment)
        
        if current_plot:
            plots.append(current_plot)
        
        return plots
    
    def _get_plot_type(self, segments: List[DataSegment]) -> str:
        """Determine plot type based on color_ids present."""
        color_ids = set(s.color_id for s in segments)
        
        if 5 in color_ids:
            return 'current_density'
        elif 6 in color_ids:
            return 'space_charge'
        else:
            return 'pierce_gun'
    
    def plot_all(self, figsize: Tuple[int, int] = (9, 16), show_grid: bool = True):
        """Create all plots with 9:16 aspect ratio."""
        plots_data = self._split_into_plots()
        figs = []
        
        for plot_idx, plot_segments in enumerate(plots_data):
            plot_type = self._get_plot_type(plot_segments)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Group by color_id
            color_groups = {}
            for segment in plot_segments:
                if segment.color_id not in color_groups:
                    color_groups[segment.color_id] = []
                color_groups[segment.color_id].append(segment)
            
            # Plot each segment
            for color_id in sorted(color_groups.keys()):
                group = color_groups[color_id]
                color = self.color_map.get(color_id, '#7f7f7f')
                label = self.legend_map.get(color_id, f'Color ID {color_id}')
                
                for idx, segment in enumerate(group):
                    show_label = label if idx == 0 else ''
                    ax.plot(segment.x_coords, segment.y_coords, 
                           color=color, 
                           linewidth=2 if color_id == 1 else 1.5,
                           alpha=1.0 if color_id == 1 else 0.7,
                           label=show_label)
            
            # Legend
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, 
                     framealpha=0.9, edgecolor='black', fancybox=True)
            
            # Axes labels based on plot type
            if plot_type == 'current_density':
                ax.set_xlabel('Z Axis', fontsize=12, fontweight='bold')
                ax.set_ylabel('Normalized Current Density', fontsize=12, fontweight='bold')
            elif plot_type == 'space_charge':
                ax.set_xlabel('Z Axis', fontsize=12, fontweight='bold')
                ax.set_ylabel('Normalized Space Charge Density at r=0', fontsize=12, fontweight='bold')
            else:
                ax.set_xlabel('Z Axis', fontsize=12, fontweight='bold')
                ax.set_ylabel('R Axis', fontsize=12, fontweight='bold')
            
            # Title
            ax.set_title('LKTRON PIERCE', fontsize=16, fontweight='bold', pad=15)
            
            if show_grid:
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Get data ranges
            all_x = [x for seg in plot_segments for x in seg.x_coords]
            all_y = [y for seg in plot_segments for y in seg.y_coords]
            
            if all_x and all_y:
                x_max = max(all_x)
                y_max = max(all_y)
                
                # Add padding only on top and right
                x_pad = x_max * 0.05
                y_pad = y_max * 0.05
                
                # Start at 0,0 (bottom-left corner)
                ax.set_xlim(0, x_max + x_pad)
                ax.set_ylim(0, y_max + y_pad)
                
                # Uniform scaling: equal units on both axes
                ax.set_aspect('equal', adjustable='box')
            
            # Footer with run number and timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            footer_text = f'Run No: {self.run_no} | Date: {timestamp}'
            fig.text(0.5, 0.01, footer_text, ha='center', fontsize=9, 
                    style='italic', color='gray')
            
            # Adjust layout: [left, bottom, right, top]
            # Leave space for title (top), xlabel and footer (bottom), legend (right)
            plt.tight_layout(rect=[0.05, 0.06, 0.85, 0.96])
            figs.append((fig, ax, plot_idx))
        
        return figs
    
    def save_plots(self, base_filename: str, dpi: int = 300):
        """Save all plots to files."""
        figs = self.plot_all()
        
        for fig, ax, idx in figs:
            if len(figs) == 1:
                filename = base_filename
            else:
                name, ext = base_filename.rsplit('.', 1)
                filename = f"{name}_plot{idx+1}.{ext}"
            
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"✓ Plot {idx+1} saved to {filename}")
            plt.close(fig)
    
    def show_plots(self):
        """Display all plots with proper aspect ratio."""
        figs = self.plot_all()
        
        # Configure matplotlib to use a backend that respects figure size
        for fig, ax, idx in figs:
            # Get current figure manager and set window size
            manager = plt.get_current_fig_manager()
            try:
                # Try to maximize or set specific window size
                manager.window.state('normal')
            except:
                pass
        
        plt.show()
    
    def save_and_show_plots(self, base_filename: str, dpi: int = 300):
        """Save and display all plots with proper aspect ratio."""
        figs = self.plot_all()
        
        # Save all plots
        for fig, ax, idx in figs:
            if len(figs) == 1:
                filename = base_filename
            else:
                name, ext = base_filename.rsplit('.', 1)
                filename = f"{name}_plot{idx+1}.{ext}"
            
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Plot {idx+1} saved to {filename}")
        
        # Display with enforced size
        for fig, ax, idx in figs:
            manager = plt.get_current_fig_manager()
            try:
                manager.resize(*manager.window.maxsize())
            except:
                pass
        
        plt.show()
    
    def print_summary(self):
        """Print detailed summary."""
        print("=" * 70)
        print("PIERCE DATA SUMMARY")
        print("=" * 70)
        print(f"Total segments: {len(self.segments)}")
        
        plots_data = self._split_into_plots()
        print(f"Number of plots: {len(plots_data)}")
        
        for plot_idx, plot_segments in enumerate(plots_data):
            plot_type = self._get_plot_type(plot_segments)
            print(f"\nPlot {plot_idx + 1} ({plot_type}):")
            
            color_counts = {}
            for segment in plot_segments:
                color_counts[segment.color_id] = color_counts.get(segment.color_id, 0) + 1
            
            for color_id in sorted(color_counts.keys()):
                label = self.legend_map.get(color_id, f'Color ID {color_id}')
                print(f"  {label}: {color_counts[color_id]} segments")
        
        print("=" * 70)


def main():
    """Main execution."""
    import sys
    
    filepath = 'pierce.plt'
    
    # Get run number from command line or use default
    run_no = sys.argv[1] if len(sys.argv) > 1 else "001"
    
    print("Parsing pierce.txt...")
    parser = PierceDataParser(filepath)
    segments = parser.parse()
    print(f"Found {len(segments)} segments\n")
    
    plotter = PiercePlotter(segments, run_no=run_no)
    plotter.print_summary()
    
    print("\nGenerating plots (9:16 aspect ratio)...")
    plotter.save_and_show_plots('pierce_plot.png', dpi=300)
    print("Complete!")


if __name__ == "__main__":
    main()