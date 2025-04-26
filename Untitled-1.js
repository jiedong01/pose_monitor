fetch('/detect_posture', {
    method: 'POST',
    body: formData
})

console.log(formData); // 检查 formData 是否包含图像数据