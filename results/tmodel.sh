
NAME=`python3 lookup_index.py $1`

echo $NAME

if [ "$1" -gt 0 ]; then
  ./slh_avg_diff.awk ~/Downloads/rlr_monthly/data/$1.rlrdata >"$NAME" 
else
  echo "No filter"
fi


env QUADEXCLUDE=$2 ../obj/index_regress "$NAME" "$3 $4" | tail -n 4 | tee metrics.txt


python3 plot.py $1 $2 $3 $4 $5

