import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

def plot_isohyet(date_str, data_path="processed_rain_data.csv", df=None):
    if df is None:
        if not os.path.exists(data_path):
            return None, f"Error: {data_path} not found."
            
        df = pd.read_csv(data_path)
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    target_date = pd.to_datetime(date_str)
    df_day = df[df['Date'] == target_date].copy()
    
    if df_day.empty:
        return None, f"No data available for date: {date_str}. Available dates range from {df['Date'].dt.date.min()} to {df['Date'].dt.date.max()}"
    
    df_day = df_day[(df_day['Lat'] != 0.0) & (df_day['Long'] != 0.0)]
    df_day = df_day.dropna(subset=['Lat', 'Long', 'Rain (inches)'])
    
    if len(df_day) < 3:
        return None, f"Not enough stations with valid Lat/Long data to interpolate for {date_str}. Found {len(df_day)} stations."
        
    x = df_day['Long'].values
    y = df_day['Lat'].values
    z = df_day['Rain (inches)'].values
    
    grid_res = 200
    xx = np.linspace(x.min() - 0.2, x.max() + 0.2, grid_res)
    yy = np.linspace(y.min() - 0.2, y.max() + 0.2, grid_res)
    xi, yi = np.meshgrid(xx, yy)
    
    method = 'cubic' if len(x) > 4 else 'linear'
    try:
        zi = griddata((x, y), z, (xi, yi), method=method, fill_value=np.nan)
        if np.isnan(zi).all():
            zi = griddata((x, y), z, (xi, yi), method='linear', fill_value=np.nan)
    except Exception as e:
        print(f"Interpolation error: {e}. Falling back to linear.")
        zi = griddata((x, y), z, (xi, yi), method='linear', fill_value=np.nan)
    
    proj = cartopy.crs.PlateCarree()
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(projection=proj)
    
    map_extent = [-123.1, -122.4, 47.4, 48.0]
    ax.set_extent(map_extent, crs=proj)
                   
    if z.max() - z.min() > 0.01:
        cs1 = plt.contourf(xi, yi, zi, cmap=plt.cm.jet, alpha=0.7, transform=proj)
        cs2 = plt.contour(xi, yi, zi, linewidths=1, colors='k', transform=proj)
        plt.clabel(cs2, fontsize=10, fmt='%1.2f', inline=True)
        cb = plt.colorbar(cs1, shrink=0.6, pad=0.05)
        cb.set_label('Rain (inches)')
    else:
        print("Note: Uniform rainfall detected, contour map might be blank.")
    
    plt.scatter(x, y, s=30, color='red', marker='o', transform=proj, zorder=5, edgecolor='black')
    
    for i, row in df_day.iterrows():
        text = f"{row['Station'].split('_')[0]}\n{row['Rain (inches)']:.2f}\""
        plt.annotate(text, (row['Long'], row['Lat']), xytext=(5, 5),
                     textcoords='offset points', fontsize=8, color='black',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                     zorder=6)
                     
    import matplotlib.image as mpimg
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KitsapMap.png")
    
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img, origin='upper', extent=map_extent, transform=proj, zorder=0)
    else:
        print(f"Warning: Background map {img_path} not found.")
    
    plt.title(f"Isohyetal Contour Map - {date_str}", loc='center', fontsize=16, pad=20)
    
    return fig, "Success"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an isohyetal contour map for a specific date.")
    parser.add_argument('--date', type=str, default='2020-01-01', help='Date to plot in format YYYY-MM-DD')
    parser.add_argument('--data', type=str, default='processed_rain_data.csv', help='Path to processed rain data CSV')
    args = parser.parse_args()
    
    fig, msg = plot_isohyet(args.date, args.data)
    if fig:
        out_name = f"isohyet_map_{args.date}.png"
        fig.savefig(out_name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Success: Isohyet map saved to {out_name}")
    else:
        print(msg)
