window.onload = () => {
	$('#sendbutton').click(() => {
		imagebox = $('#bg')
		input = $('#imageinput')[0]
		console.log("abc")
		if(input.files && input.files[0])
		{
			let formData = new FormData();
			formData.append('image' , input.files[0]);
			$.ajax({
				url: "http://localhost:5000/test", // fix this to your liking
				type:"POST",
				data: formData,
				cache: false,
				processData:false,
				contentType:false,
				error: function(data){
					console.log("upload error" , data);
					console.log(data.getAllResponseHeaders());
				},
				success: function(data){
					// alert("hello"); // if it's failing on actual server check your server FIREWALL + SET UP CORS
					bytestring = data['status']
					image = bytestring.split('\'')[1]
					imagebox.attr('src' , 'data:image/jpeg;base64,'+image)
				}
			});
			console.log("worked")
		}
	});
};



function readUrl(input){
	imagebox = $('#bg')
	console.log("evoked readUrl")
	if(input.files && input.files[0]){
		let reader = new FileReader();
		reader.onload = function(e){
			console.log(e)
			
			imagebox.attr('src',e.target.result); 
		}
		reader.readAsDataURL(input.files[0]);
	}

	
}