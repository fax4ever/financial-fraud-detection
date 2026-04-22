## Preliminary steps

0. Environment variables and login

> oc login --token=...

```bash
export NAMESPACE=fraud-detection
export NGC_API_KEY=<your-ngc-key>
```

1. Delete / create a project

> oc delete project $NAMESPACE
> oc new-project $NAMESPACE
> oc adm policy add-scc-to-user anyuid -z default -n $NAMESPACE

2. Install preliminary resources

> ./openshift/deploy-init-pipeline.sh

```bash
fax@fercoli-mac financial-fraud-detection % oc get pvc,deployment,svc,secret,configmap -l app.kubernetes.io/name=fraud-init-pipeline -n "$NAMESPACE"
NAME                                 STATUS    VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   VOLUMEATTRIBUTESCLASS   AGE
persistentvolumeclaim/fraud-data     Pending                                                                        gp3-csi        <unset>                 10m
persistentvolumeclaim/fraud-models   Pending                                                                        gp3-csi        <unset>                 10m
persistentvolumeclaim/minio-pvc      Bound     pvc-00a7528a-6a85-4942-8af7-6ddfaa099a0b   20Gi       RWO            gp3-csi        <unset>                 10m

NAME                    READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/minio   1/1     1            1           10m

NAME            TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)             AGE
service/minio   ClusterIP   172.30.45.217   <none>        9000/TCP,9001/TCP   10m

NAME                  TYPE                             DATA   AGE
secret/minio-secret   Opaque                           2      10m
secret/ngc-secret     kubernetes.io/dockerconfigjson   1      10m

NAME                                  DATA   AGE
configmap/fraud-kfp-training-config   1      10m
```

## Deploy the pipeline using RedHat OpenShift AI

It is based on the following documentation: https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html-single/working_with_ai_pipelines/index

1. From the OpenShift AI dashboard, click `Projects` and then selecting `fraud-detection` project.

2. Click `Pipelines` tab, and then `Configure pipeline server` with the following parameters:

| Field | Value |
|---|---|
| **Bucket** | `fraud-pipelines` |
| **Access key** | `minio` |
| **Secret key** | `minio123` |
| **Endpoint** | `http://minio.fraud-detection.svc:9000` |
| **Region** | leave empty |

Pipeline server will be deployed as a series of pods in the current $NAMESPACE.

```bash
oc get pods -n "$NAMESPACE" -w
```

3. Click on `Import pipeline`, passing [fraud_kfp_configmap.yaml](./pipeline/fraud_kfp_configmap.yaml) in `Upload file`, and `fraud-preprocess-train` as `Pipeline name`.

## Run the pipeline

Run exotic commands like:

oc adm policy add-scc-to-user anyuid -z pipeline-runner-dspa -n "$NAMESPACE"

oc run fix-perms --rm -it --restart=Never --image=busybox --overrides='{"spec":{"securityContext":{"runAsUser":0},"containers":[{"name":"fix","image":"busybox","command":["chmod","-R","777","/data"],"volumeMounts":[{"name":"d","mountPath":"/data"}]}],"volumes":[{"name":"d","persistentVolumeClaim":{"claimName":"fraud-data"}}]}}'

Create the pipeline instance and the run instance, then run it, as `Data science pipeline`.

