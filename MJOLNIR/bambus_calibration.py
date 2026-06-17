bambus_file = "CalibrationBambus.csv"
angle_increment = 2
n_wedges = 20
detectors_per_wedge = 5
pix = 0
amp = 1
Ef = (3.0, 3.5, 4.0, 4.5, 5.0)
width = 1
bkg = 0
lbin = 0
ubin = 1
header = "  detector,    energy,     pixel, amplitude,    center,     width,background,  lowerBin,  upperBin,  A4Offset\n"
with open(bambus_file, 'w') as outfile:
    outfile.write(header)

count = 0;
for wedge_idx in range(n_wedges):
    for det_idx in range(detectors_per_wedge):
        a4 = 19.0 - wedge_idx * angle_increment
        a4offset = a4
        line = f"{wedge_idx:>10d},{det_idx:>10d},{pix:>10d},{amp:>10.4f},{Ef[det_idx]:>10.4f},{width:>10.4f},{bkg:>10.4f},{lbin:>10.4f},{ubin:>10.4f},{a4offset:>10.4f}\n"
        with open(bambus_file, 'a') as outfile:
            outfile.write(line)

