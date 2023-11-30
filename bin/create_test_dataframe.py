from glob import glob

import numpy as np
import pandas as pd

from ml_aos.dataloader import Donuts

# load the test donuts
donuts = Donuts("test", transform=False)
data_dir = donuts.settings["data_dir"]
obs_table = donuts.observations

# get the unique list of pointings
pointings = np.unique(
    [int(file.split("/")[-1].split(".")[0][3:]) for file in donuts.image_files["test"]]
)

# loop through the pointings, load the data, and save in dataframe
meta = []
blend_mags = []
blend_offsets = []
images = []
zernikes = []
dofs = []

for pntId in pointings:
    # get image file names
    image_files = glob(f"{data_dir}/images/pnt{pntId}.*")

    # get the observation row
    obsId = int(image_files[0].split("/")[-1].split(".")[1][3:])
    obs_row = obs_table[obs_table["observationId"] == obsId]

    # pull out observation data
    band = obs_row["lsstFilter"].item()
    airmass = obs_row["airmass"].item()
    seeing = obs_row["seeingFwhm500"].item()
    sky = obs_row["skyBrightness"].item()

    # and the catalog
    catalog = pd.read_parquet(f"{data_dir}/catalogs/pnt{pntId}.catalog.parquet")

    for file in image_files:
        # get the object ID
        objId = int(file.split("/")[-1].split(".")[2][3:])

        # select the object and blends
        cat_row = catalog[catalog["objectId"] == objId]

        # get info about object
        ra, dec = np.rad2deg([cat_row["ra"].item(), cat_row["dec"].item()])
        fx, fy = np.rad2deg([cat_row["xField"].item(), cat_row["yField"].item()])
        chip = cat_row["detector"].item()
        corner = chip[:3]
        intra = int(chip[-1]) == 1
        mag = cat_row["lsstMag"].item()
        temp = cat_row["temperature"].item()

        # get info about blends
        blends = catalog[
            (catalog["blendId"] == objId) & (catalog["aosSource"] == False)
        ]
        mags = blends["lsstMag"].to_numpy()
        offsets = np.array(
            [
                (blends["xCentroid"] - cat_row["xCentroid"].item()).to_numpy(),
                (blends["yCentroid"] - cat_row["yCentroid"].item()).to_numpy(),
            ]
        )

        # only flag blends that are within 130 pixels
        radii = np.sqrt(np.square(offsets).sum(axis=0))
        mags = mags[radii < 130]
        offsets = offsets[:, radii < 130]

        # correct offsets for CWFS rotation
        if chip in ["R00_SW1", "R44_SW0"]:
            offsets = offsets
        elif chip in ["R40_SW1", "R04_SW0"]:
            offsets = np.roll(offsets, 1, axis=0) * np.array([+1, -1])[:, None]
        elif chip in ["R44_SW1", "R00_SW0"]:
            offsets = np.roll(offsets, 1, axis=0) * np.array([-1, -1])[:, None]
        elif chip in ["R04_SW1", "R40_SW0"]:
            offsets = np.roll(offsets, 1, axis=0) * np.array([-1, +1])[:, None]

        meta.append(
            [
                pntId,
                obsId,
                objId,
                ra,
                dec,
                fx,
                fy,
                corner,
                intra,
                mag,
                band,
                seeing,
                airmass,
                sky,
            ]
        )
        blend_mags.append(mags)
        blend_offsets.append(offsets)
        images.append(np.load(file)[5:-5, 5:-5])
        zernikes.append(
            np.load(
                f"{data_dir}/zernikes/pnt{pntId}.obs{obsId}.detector{corner}.zernikes.npy"
            )
        )
        dofs.append(np.load(f"{data_dir}/dof/pnt{pntId}.dofs.npy"))


# convert the metadata into a dataframe
df = pd.DataFrame(
    meta,
    columns=[
        "pntId",
        "obsId",
        "objId",
        "ra",
        "dec",
        "fx",
        "fy",
        "corner",
        "intra",
        "mag",
        "filter",
        "seeing",
        "airmass",
        "skyBrightness",
    ],
)

# add columns for other data
df["blendMags"] = blend_mags
df["blendOffsets"] = blend_offsets
df["image"] = images
df["zernikes"] = zernikes
df["dof"] = dofs


def flag_bright_blends(
    row: pd.Series,
    isoMagDiff: float = 2,
):
    """Flag bright blends that we should mask."""
    brightBlends = row["blendMags"] <= row["mag"] + isoMagDiff

    # save list of offsets for bright blends
    row["brightBlendOffsets"] = row["blendOffsets"][:, brightBlends]

    return row


df = df.apply(select_blends, axis=1)

# save the dataframe!
df.to_pickle("../data/test_dataframe.pkl")
