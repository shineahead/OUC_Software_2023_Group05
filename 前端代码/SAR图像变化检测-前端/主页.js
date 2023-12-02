document.getElementById('inputGroupFile01').addEventListener('change', function (e) {
    var file = e.target.files[0];  // 获取选中的文件  
    var reader = new FileReader();  // 创建一个FileReader对象  

    reader.onloadend = function () {
        // console.log(reader.result)
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

var cnt = 1
document.getElementById("generate_result").addEventListener("click", function (e) {
    var base64_1 = document.getElementById('imagePreview1').src
    var base64_2 = document.getElementById('imagePreview2').src

    var substr1 = base64_1.substring(0, 10)
    var substr2 = base64_2.substring(0, 10)

    if (substr1 == "data:image" && substr2 == "data:image") {
        console.log("非空")
        var arr = [base64_1, base64_2]
        //把两张图片的baseURL传入到后端
        axios.post(
            "http://127.0.0.1:8000/v1/api/SARDetection/",
            {
                body: "axios post function send",
                data: JSON.stringify(arr),
                postId: cnt
            }
        ).then(response => {
            var code = response.status
            if (code == 201) console.log("发送成功！")
            var base = response.data.base
            console.log(base)
            // 设置检测图片的SRC
            document.getElementById('resImage').src = base

            cnt += 1
        })

    } else {
        alert("请先上传图片!")
    }
})
