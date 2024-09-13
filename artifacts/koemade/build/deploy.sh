cd ../../
tar --exclude='./koemade/uploads' -cf koemade.tar ./koemade/
scp koemade.tar altneues@s16.valueserver.jp:~/
ssh altneues@s16.valueserver.jp '
tar -xf koemade.tar &&
if [ -d ./public_html/stg2.koemade.net/uploads ]; then
  mv ./public_html/stg2.koemade.net/uploads ./koemade/uploads
fi &&
rm -rf ./public_html/stg2.koemade.net &&
mv ./koemade/ ./public_html/stg2.koemade.net
'
