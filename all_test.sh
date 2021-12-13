test_name=$1
samples=$2

if [ -z "$2" ]
    then
        echo -e 'running full datasets'
        samples=1000000
fi

echo -e '\nrunning input complexity test \n'
echo -e '------------------------------\n'
python complexity.py --test_name $test_name --samples $samples
echo -e '\nrunning likelihood ratio test \n'
echo -e '------------------------------\n'
python ratio.py --test_name $test_name --samples $samples
echo -e '\nrunning likelihood regret test \n'
echo -e '------------------------------\n'
python regret.py  --test_name $test_name --samples $samples

