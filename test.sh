TIMES=10
echo "$TIMES learning"

counter=0
echo > ./outputs/DK_test

for i in {1..10}
do
    echo "$i unsupervised_learning.py"
    python3 unsupervised_learning.py > ./outputs/DK_test

done

echo "done"
