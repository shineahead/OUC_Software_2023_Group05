document.getElementById('inputGroupFile01').addEventListener('change', function (e) {
    var file = e.target.files[0];  // 获取选中的文件  
    var reader = new FileReader();  // 创建一个FileReader对象  

    reader.onloadend = function () {
        document.getElementById('imagePreview1').src = reader.result;  // 将预览图片的src设为读取到的数据  
    }

    if (file) {
        reader.readAsDataURL(file);  // 读取文件内容作为URL  
    } else {
        // 处理没有文件选择的情况  
    }
});

document.getElementById('inputGroupFile02').addEventListener('change', function (e) {
    var file = e.target.files[0];  // 获取选中的文件  
    var reader = new FileReader();  // 创建一个FileReader对象  

    reader.onloadend = function () {
        document.getElementById('imagePreview2').src = reader.result;  // 将预览图片的src设为读取到的数据  
    }

    if (file) {
        reader.readAsDataURL(file);  // 读取文件内容作为URL  
    } else {
        // 处理没有文件选择的情况  
    }
});