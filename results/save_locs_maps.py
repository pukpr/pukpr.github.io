import sys
import csv
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def convert_latlon(lat, ns, lon, ew):
    lat = float(lat)
    lon = float(lon)
    if ns.upper() == 'S':
        lat = -lat
    if ew.upper() == 'W':
        lon = -lon
    return lat, lon

def plot_location(fields):
    # Parse the required fields
    try:
        ident = fields[0]
        name = fields[1].strip('"')
        lat = fields[2]
        ns = fields[3]
        lon = fields[4]
        ew = fields[5]
        lat, lon = convert_latlon(lat, ns, lon, ew)
    except Exception as e:
        print(f"Error parsing fields: {e}")
        return

    # Set up the map
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='white')

    # Plot the point
    ax.plot(lon, lat, 'ro', markersize=7, transform=ccrs.PlateCarree())
    ax.text(lon, lat, name, fontsize=10, transform=ccrs.PlateCarree(),
            ha='left', va='bottom', color='blue')

    plt.title("Location: " + name)
    plt.tight_layout()
    plt.savefig(ident+'_loc.png')  # Save as PNG 
    # plt.show()

def lookup_site_name(csv_file):
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            plot_location(row)

if __name__ == "__main__":
    lookup_site_name(sys.argv[1])
