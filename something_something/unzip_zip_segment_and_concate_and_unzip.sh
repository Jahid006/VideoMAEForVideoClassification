for i in {00..19}; do
    unzip zip_only/20bn-something-something-v2-"$i".zip
done

cat 20bn-something-something-v2-?? >> 20bn-something-something-v2.tar.gz
tar -xvzf 20bn-something-something-v2.tar.gz

