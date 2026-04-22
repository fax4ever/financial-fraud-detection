# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Kubeflow Pipeline: TabFormer preprocess (fraud-data PVC) then GNN training
# (fraud-data + fraud-models + training config).
#
# Prereqs in the target namespace:
#   - PVCs: fraud-data, fraud-models
#   - ConfigMap: fraud-kfp-training-config
#   - Secret: ngc-secret for nvcr.io (see openshift/deploy-training.sh)
#
# Compile (from this directory, with kfp and kfp-kubernetes installed):
#   cd openshift/pipeline && uv run python fraud_preprocess_train_pipeline.py
# Upload the generated .yaml in the Kubeflow Pipelines UI or with the kfp CLI.
from kfp import compiler, dsl, kubernetes

_DEFAULT_PREPROCESS = "quay.io/fercoli/fraud-preprocess:latest"
_DEFAULT_TRAIN = "nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0"


@dsl.component(base_image=_DEFAULT_PREPROCESS)
def run_preprocess():
    import os
    import shutil
    import sys

    sys.path.insert(0, "/workspace/src")
    print("Setting up symlink...", flush=True)
    raw_dir = "/data/TabFormer/raw"
    link = os.path.join(raw_dir, "card_transaction.v1.csv")
    os.makedirs(raw_dir, exist_ok=True)
    if os.path.exists(link) or os.path.islink(link):
        os.remove(link)
    os.symlink("/workspace/raw/card_transaction.v1.csv", link)
    print("Starting preprocessing...", flush=True)
    from preprocess_TabFormer_lp import preprocess_data

    preprocess_data("/data/TabFormer")
    print("Cleaning up...", flush=True)
    os.remove(link)
    shutil.rmtree(raw_dir)


@dsl.container_component
def run_training(training_image: str) -> dsl.ContainerSpec:
    # str() works around kubeflow/pipelines#10657: protobuf rejects the raw
    # InputValuePlaceholder object; casting emits the placeholder string that
    # the backend resolves at runtime.
    return dsl.ContainerSpec(
        image=str(training_image),
        command=["/opt/nvidia/nvidia_entrypoint.sh", "bash", "-c",
                 "cp /config/config.json /app/config.json && python /app/main.py --config /app/config.json"],
    )


@dsl.pipeline(name="fraud-preprocess-train")
def fraud_preprocess_train(
    training_image: str = _DEFAULT_TRAIN,
    fraud_data_pvc: str = "fraud-data",
    fraud_models_pvc: str = "fraud-models",
    training_configmap: str = "fraud-kfp-training-config",
    ngc_pull_secret: str = "ngc-secret",
) -> None:
    preprocess: dsl.PipelineTask = run_preprocess()
    train: dsl.PipelineTask = run_training(training_image=training_image)

    kubernetes.set_image_pull_secrets(train, [ngc_pull_secret])
    kubernetes.set_image_pull_policy(preprocess, "Always")
    kubernetes.set_image_pull_policy(train, "IfNotPresent")

    kubernetes.set_timeout(preprocess, 3 * 3600)
    kubernetes.set_timeout(train, 4 * 3600)
    preprocess.set_caching_options(False)
    train.set_caching_options(False)
    train.after(preprocess)

    for task in [preprocess, train]:
        task.set_gpu_limit("1")
        task.set_memory_request("16Gi").set_memory_limit("48Gi")
        task.set_cpu_request("4").set_cpu_limit("8")
        kubernetes.add_toleration(task, key="g6-gpu", operator="Exists", effect="NoSchedule")
        kubernetes.add_toleration(task, key="p4-gpu", operator="Exists", effect="NoSchedule")
        kubernetes.add_node_selector(task, "nvidia.com/gpu.present", "true")

    kubernetes.mount_pvc(
        task=preprocess,
        pvc_name=fraud_data_pvc,
        mount_path="/data",
    )
    kubernetes.mount_pvc(
        task=train,
        pvc_name=fraud_data_pvc,
        mount_path="/data",
    )
    kubernetes.mount_pvc(
        task=train,
        pvc_name=fraud_models_pvc,
        mount_path="/trained_models",
    )
    kubernetes.use_config_map_as_volume(
        task=train,
        config_map_name=training_configmap,
        mount_path="/config",
    )


if __name__ == "__main__":
    out = __file__.replace(".py", ".yaml")
    compiler.Compiler().compile(
        pipeline_func=fraud_preprocess_train,
        package_path=out,
    )
    print(f"Wrote {out}")
