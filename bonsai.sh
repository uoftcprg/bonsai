sfps=$1
us=$2
sfps2=$3
us2=$4
maps=$5
maps2=$(mktemp --suffix .npz)
rounds=0

./trim.py $sfps $us $sfps2 $us2 $maps

while
	stat=$(./stat.py $sfps2 $us2)

	./graph.py $sfps2 $us2 | ./refine.py | ./fold.py $sfps2 $us2 $sfps2 $us2 $maps2
	./compose.py $maps $maps2 $maps
	./trim.py $sfps2 $us2 $sfps2 $us2 $maps2
	./compose.py $maps $maps2 $maps

	[ "$stat" != "$(./stat.py $sfps2 $us2)" ]
do
	rounds=$((rounds + 1))
done

rm $maps2
echo $rounds
