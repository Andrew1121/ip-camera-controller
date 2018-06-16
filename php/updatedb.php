<?php
  include '../php/connectdb.php';

  $top = $_POST['top'];
  $left = $_POST['left'];
  $right = $_POST['right'];
  $bottom = $_POST['bottom'];

  $sql = "UPDATE camerapos SET toppos = ".$top." ,leftpos = ".$left.", rightpos = ".$right.", bottompos = ".$bottom." WHERE id='currentpos'";

  if ($conn->query($sql) === TRUE) {
      echo "Record updated successfully";
  } else {
      echo "Error updating record: " . $conn->error;
  }

?>
