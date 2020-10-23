#! /bin/bash
# Expected to be launched with DATA_DIR as first argument

set -ex # stop and fail if anything stops

echo "Extracting all..."
cd $1

extract_and_delete() {
    filename=$1
    OUTDIR=${filename%.tar}
    tar -xf $filename --xform="s|^|$OUTDIR/|S"
    rm $filename    
}

for filename in train/*.tar; do
    extract_and_delete $filename &
done

wait

cd val
# https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md
tar -xf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
echo "Done!"
cd ..
echo "Extracted `find train | grep .JPEG | wc -l` train files and `find val | grep .JPEG | wc -l` validation files"

