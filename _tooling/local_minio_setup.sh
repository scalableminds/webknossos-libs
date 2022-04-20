source "$(dirname "${BASH_SOURCE[0]}")/trap_add.sh"

export MINIO_ROOT_USER="TtnuieannGt2rGuie2t8Tt7urarg5nauedRndrur"
export MINIO_ROOT_PASSWORD="ANTN35UAENTS5UIAEATD"

# Minio is an S3 clone and is used as local test server
docker run \
  -p 8000:9000 \
  -e MINIO_ROOT_USER=$MINIO_ROOT_USER \
  -e MINIO_ROOT_PASSWORD=$MINIO_ROOT_PASSWORD \
  --name minio \
  --rm \
  -d \
  minio/minio server /data

stop_minio () {
    docker stop minio
}
trap_add stop_minio EXIT
