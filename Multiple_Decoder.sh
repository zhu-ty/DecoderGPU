printf "" > tmp.txt
count=0;
#echo $count
for i in `ls *.bin`
do
    b=$(( $count % 2 ))
    if [ $b = 0 ]; then
        #echo 0;
        #echo "$i 0\n" >> tmp.txt
        printf "./DecoderGPU $i 0 20\n" >> tmp.txt
    else
        #echo 1;
        printf "./DecoderGPU $i 1 20\n" >> tmp.txt
    fi
    #((count++));
    count=`expr $count + 1`
done

#cat tmp.txt | xargs -L 1 -P 4 -I {} echo {} | cut -d' ' -f1,2 | xargs -n 2 sh -c './DecoderGPU $0 $1'

cat tmp.txt | xargs -L 1 -P 2 -I xiaoyun sh -c "xiaoyun"

rm tmp.txt
