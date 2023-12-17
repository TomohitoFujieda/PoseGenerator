function reshapeCoordinatesToDic(width, height, coordinates) {
    const target_idx = [1, 0, 2, 6, 8, 3, 7, 9, 4, 10, 12, 5, 11, 13, 14, 15, 16, 17];
    const keypoints = target_idx.map(i => coordinates[i]);
    const res = {
        "width": width,
        "height": height,
        "keypoints": keypoints
    };
    return res;
}

function downloadJSON(posedata) {
    // const width = {{WIDTH|tojson}};
    // const height = {{HEIGHT|tojson}};
    // const coordinates = {{generated_keypoints|tojson}};
    // const posedata = reshapeCoordinatesToDic(width, height, coordinates);
    const json = JSON.stringify(posedata);
    const blob = new Blob([json], {type: "application/json"});
    const filename = "generated-pose-" + Date.now().toString() + ".json";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
}

function downloadPNG(url) {
    const filename = "generated-poseimg-" + Date.now().toString() + ".png";
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
}