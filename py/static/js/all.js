$(function(){
  var req, pan = 0, tilt = 0;
  var duration, initDur;
  var initTimer = 0;
  var angPan, angTilt;

  updateImg();

  //send request to IP Camera
  var ctrlUrl="http://123.203.187.166:80/cgi-bin/api.cgi?user=root&password=123456&cmd=PtzCtrl&token=141263da251ab9e";
  var json;
  var initHandler = {
    init: function(){
      var orgx, orgy;
      $(".loading-overlay").show();
      makeCtrlRequest("RightDown");
      var init = setInterval(function(){
        initTimer++;
        if(initTimer == 22){
          clearInterval(init);
          makeCtrlRequest("Stop");
          updateImg();
          calcPan(pan);
          calcTilt(tilt);
          $(".loading-overlay").hide();
          // selectRequest();
        }
      }, 1000);
    }
  }
  var controlHandler = {
    init: function(){
      $('#btn-up').on('mousedown', function(e) {
        e.preventDefault();
        makeCtrlRequest("Up");
        duration = setInterval(function(){
          if(tilt >= 13){
            console.log(tilt);
            clearInterval(duration);
            $(this).trigger('mouseup');
          }else{
            tilt++;
          }
        }, 1000);
      }).on('mouseup', function() {
        stopRequest()
        calcTilt(tilt);
      });
      $('#btn-down').on('mousedown', function(e) {
        e.preventDefault();
        makeCtrlRequest("Down");
        duration = setInterval(function(){
          if(tilt <= 0){
            clearInterval(duration);
            $(this).trigger('mouseup');
          }else{
            tilt--;
          }
        }, 1000);
      }).on('mouseup', function() {
        stopRequest()
        calcTilt(tilt);
      });
      $('#btn-left').on('mousedown', function(e) {
        e.preventDefault();
        makeCtrlRequest("Left");
        duration = setInterval(function(){
          if(pan >= 16){
            clearInterval(duration);
            $(this).trigger('mouseup');
          }else{
            pan++;
          }
        }, 1000);
      }).on('mouseup', function() {
        stopRequest()
        calcPan(pan);
      });
      $('#btn-right').on('mousedown', function(e) {
        e.preventDefault();
        makeCtrlRequest("Right");
        duration = setInterval(function(){
          if(pan <= 0){
            clearInterval(duration);
            $(this).trigger('mouseup');
          }else{
            pan--;
          }
        }, 1000);
      }).on('mouseup', function() {
        stopRequest();
        calcPan(pan);
      });
      $('#btn-zoom1').click(function(e) {
        e.preventDefault();
        $("#video-stream").css("background-size","100%");
      });
      $('#btn-zoom15').click(function(e) {
        e.preventDefault();
        $("#video-stream").css("background-size","150%");
      });
      $('#btn-zoom20').click(function(e) {
        e.preventDefault();
        $("#video-stream").css("background-size","200%");
      });
    }
  }

  // $('#btn-test').click(function(e){
  //   e.preventDefault();
  //   json = [{"cmd":"PtzCtrl","action":0,"param":{"channel":0,"op":"ToPos","speed":32,"id":1}}];
  //   makeCtrlRequest();
  // });

  function makeCtrlRequest(param){
    //create request for shopify
    if(param=="Stop"){
      json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Stop"}}];
    }else{
      json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": param, "speed": 32}}];
    }

    req = new XMLHttpRequest();

    req.onreadystatechange=function() {
      if(req.readyState==4) {
        var showErrorTab=false;
      }
    }

    req.open("POST",ctrlUrl);
    req.setRequestHeader("Accept","application/json");
    req.send(JSON.stringify(json));
  }

  function updateRecord(){
    // var data = {id:'currentpos', pan: pan, tilt: tilt};
    // $.post('../php/updatedb.php', data, function(returnedData){
    //   console.log(returnedData);
    // });
    updateImg();
  }

  // function selectRequest(){
  //   $.ajax({    //create an ajax request to selectdb.php
  //       type: "POST",
  //       url: "../php/selectdb.php",
  //       dataType: "json",
  //       success: function(data){
  //         pan = data.result.pan;
  //         tilt = data.result.tilt;
  //         calcPan(pan);
  //         calcTilt(tilt);
  //       },
  //       error: function (xhr, ajaxOptions, thrownError) {
  //         alert(thrownError);
  //       }
  //   });
  //
  // }

  function stopRequest(){
    clearInterval(duration);
    makeCtrlRequest("Stop");
    updateRecord();
  }

  function calcPan(t){
    angPan = Math.ceil(t/16 * 355);
    $("#pan").html(angPan);
  }

  function calcTilt(t){
    angTilt = Math.ceil(t/13 * 105)-105;
    $("#tilt").html(angTilt);
  }

  // function initPos(){
  //   var tmp = tilt;
  //
  //   if(tilt > 0){
  //     makeCtrlRequest("Up");
  //     while(tmp > 0){
  //
  //     }
  //   }
  //   var timer = tilt * 1000;
  //   setTimeout(function(){
  //     if(pan > 0){
  //       tmp = 0;
  //       makeCtrlRequest("Left");
  //       initDur = setInterval(function(){
  //         if(tmp > pan){
  //           clearInterval(initDur);
  //           makeCtrlRequest("Stop");
  //         }else{
  //           tmp++;
  //         }
  //       }, 1000);
  //     }
  //     setTimeout(function(){
  //       updateImg();
  //       $(".loading-overlay").hide();
  //     }, 15000);
  //   }, timer);
  //
  // }

  function updateImg(){
    setTimeout(function(){
      var url = "http://123.203.187.166:80/cgi-bin/api.cgi?cmd=Snap&channel=0&user=root&password=123456" + '&'+ new Date().getTime();
      $('#video-stream').css('background-image', 'url(' + url + ')');
    }, 1000);
  }

  controlHandler.init();
  initHandler.init();

});
