$(document).ready(function () {
  $('.sidenav').sidenav();
  $('#response-container').hide();
  $.ajax({
    type: 'POST',
    url: "ton_endpoint_post",
    data: formData,
    contentType: false,
    cache: false,
    processData: false,
    success: function (response) {
      $('#response-container').show();
      console.log(response);
    },
    error: function (xhr, resp, text) {
      let response = JSON.parse(xhr.responseText);
      console.log(xhr, resp, text);
    }
  });

});
