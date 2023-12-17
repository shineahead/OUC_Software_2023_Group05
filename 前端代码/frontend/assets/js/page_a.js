// 设置一个标志位，表示弹窗是否已经显示
var popupDisplayed = false;
var popup = document.getElementById("popup")
var navlink1 = document.getElementById("link1")
var navlink2 = document.getElementById("link2")
var navlink3 = document.getElementById("link3")
var navlink4 = document.getElementById("link4")
var navlink5 = document.getElementById("link5")
var navlink6 = document.getElementById("link6")
// 设置滚动监听器
window.addEventListener("scroll", function () {
    // 获取滚动的垂直距离
    var scrollDistance = window.scrollY;

    // 设置一个滚动距离的阈值，例如 200 像素
    var scrollThreshold = 900;

    // 判断滚动距离是否超过阈值且弹窗未显示
    if (scrollDistance > scrollThreshold && !popupDisplayed) {
        popup.style.display = "block";
        setTimeout(function () {
            popup.style.transform = "translate(-50%, -50%) scale(1)";
        }, 0);
        popupDisplayed = true;
    }
    if (scrollDistance > scrollThreshold - 20) {
        navlink1.style.color = "black";
        navlink2.style.color = "black";
        navlink3.style.color = "black";
    }
    else {
        navlink1.style.color = "#ffffffb6";
        navlink2.style.color = "#ffffffb6";
        navlink3.style.color = "#ffffffb6";
    }
});

document.getElementById("close_popup_button").addEventListener("click", function () {
    popup.style.display = "none";
})

document.getElementById('nav-button').addEventListener("click", function () {
    window.scrollTo({
        top: 960,
        behavior: 'smooth'
    })

})

let scale_row = 0; // 初始化进度条长度（防止用户多开卡bug，所以要设为全局变量）


let intervalId; // 定时器

function openPopup() {
    navlink1.setAttribute('href', 'javascript:void(0)');
    navlink2.setAttribute('href', 'javascript:void(0)');
    navlink3.setAttribute('href', 'javascript:void(0)');
    navlink4.setAttribute('href', 'javascript:void(0)');
    navlink5.setAttribute('href', 'javascript:void(0)');
    navlink6.setAttribute('href', 'javascript:void(0)');
    document.getElementById("tanchuang").style.display = "block";
    intervalId = setInterval(() => {
        if (scale_row < 98) {
            scale_row = scale_row + 1; // 每次执行时增加计数器变量的值
        }
        console.log(scale_row);
        document.getElementById("barlen").style.width = scale_row + '%';
    }, 2400);
}

function closePopup() {
    navlink1.setAttribute('href', './page_a.html');
    navlink2.setAttribute('href', './page_b.html');
    navlink3.setAttribute('href', './page_c.html');
    navlink4.setAttribute('href', './page_a.html');
    navlink5.setAttribute('href', './page_b.html');
    navlink6.setAttribute('href', './page_c.html');
    document.getElementById("tanchuang").style.display = "none";
    scale_row = 0;
    document.getElementById("barlen").style.width = '0%';
    clearInterval(intervalId);
}

document.getElementById('inputGroupFile01').addEventListener('change', function (e) {
    var file = e.target.files[0];  // 获取选中的文件  
    var reader = new FileReader();  // 创建一个FileReader对象  

    reader.onloadend = function () {
        document.getElementById('imagePreview1').src = reader.result;  // 将预览图片的src设为读取到的数据  
    }

    if (file) {
        reader.readAsDataURL(file);  // 读取文件内容作为URL
        console.log(file)
        // console.log(reader.result)
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
        openPopup();
        //把两张图片的baseURL传入到后端
        axios.post(
            "http://124.70.219.54/v1/api/SARDetection/",
            {
                body: "axios post function send",
                data: JSON.stringify(arr),
                postId: cnt
            }
        ).then(response => {
            console.log(response)
            var code = response.status
            if (code == 201) console.log("发送成功！")
            var base = response.data.base
            console.log(base)
            // 设置检测图片的SRC
            document.getElementById('result_image').src = base

            cnt += 1
            scale_row = 100;
            document.getElementById("barlen").style.width = scale_row + '%';
        })

    } else {
        alert("请先上传图片!")
    }
})

document.getElementById('btn_close').onclick = function () {
    closePopup();
}

function downloadImage() {
    let imgElement = document.getElementById('result_image');
    let imageUrl = imgElement.getAttribute('src'); // 要下载的图片URL
    let imageName = 'result.jpg'; // 要下载的文件名

    // 模拟创建一个<a>，模拟点击该链接并下载图片，最后再删除这个临时标签
    let a = document.createElement('a');
    a.href = imageUrl;
    a.download = imageName;
    a.style.display = 'none';
    a.target = "_blank";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

document.getElementById('btn_download').onclick = function () {
    downloadImage();
}


