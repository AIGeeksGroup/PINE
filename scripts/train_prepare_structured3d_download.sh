#!/usr/bin/env bash
# Download Structured3D perspective (furnished) zips: bash THIS OUTPUT_DIR STREAMS ZIP_ID...  (e.g. 01 02 ... 17)
# Zip k holds scenes 200k..200k+199; zip 09 is broken upstream (github.com/bertjiazheng/Structured3D/issues/30).
# Each line of the report is: zip id, bytes on disk, bytes on the server.
url=https://zju-kjl-jointlab-azure.kujiale.com/Structured3D
mkdir -p "$1"
cd "$1"
printf '%s\n' "${@:3}" | xargs -P "$2" -I{} curl -s -C - --retry 10 --retry-all-errors \
    -o Structured3D_perspective_full_{}.zip "$url/Structured3D_perspective_full_{}.zip"
for id in "${@:3}"; do
    server=$(curl -sI "$url/Structured3D_perspective_full_$id.zip" | awk 'tolower($1) == "content-length:" {print $2}' | tr -d '\r')
    echo "$id $(stat -c %s "Structured3D_perspective_full_$id.zip") $server"
done
