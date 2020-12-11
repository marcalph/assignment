$(document).ready(function () {
  $('.sidenav').sidenav();
  $('#response-container').hide();
  // predict
  $( "#predict" ).click(function() {
  var input = $("#input").val();
  input = input.split(',');
    $.ajax({
      type: 'POST',
      url: "http://localhost:8000/predict",
      data: JSON.stringify(
        {
           'fake_opportunity_id':     		 			                            input[0] ,                                   
           'opportunity_creation_date':     		 		                      	input[1] ,                                                   
           'fake_contact_id':     			 			                              input[2] ,                                  
           'contact_creation_date':     		 			                          input[3] ,                                                   
           'country':    	 						                                      input[4] ,                                  
           'salesforce_specialty':     						                          input[5] ,                                                   
           'contact_status':     						                                input[6] ,                                        
           'current_opportunity_stage':     					                      input[7] ,                                      
           'opportunity_stage_at_the_time_of_creation': 			              input[8] ,                                           
           'opportunity_stage_after_60_days':     				                  input[9] ,                                  
           'opportunity_stage_after_90_days':     				                  input[10],                                   
           'previous_max_stage':     						                            input[11],                                         
           'count_previous_opportunities':     					                    input[12],                                
           'has_mobile_phone':     						                              input[13],                                    
           'main_competitor':     						                              input[14],                                                 
           'has_website':   	  						                                input[15],                                    
           'pms':     								                                      input[16],                                        
           'pms_status':     							                                  input[17],                                       
           'count_total_calls':     						                            input[18],                                
           'count_unsuccessful_calls':     					                        input[19],                                
           'count_total_appointments':     					                        input[20],                                
           'count_contacts_converted_last_30d':     	 			                input[21],                                   
           'count_contacts_converted_last_30d_per_specialty': 		 	        input[22],                                  
           'count_contacts_converted_last_30d_per_zipcode':     		        input[23],                                
           'count_contacts_converted_last_30d_per_specialty_and_zipcode': 	input[24],                                
           'gender':     							                                      input[25],                                  
           'practitioner_age':     						                              input[26],                                   
           'years_since_graduation':     					                          input[27],                                   
           'years_since_last_moving':     					                        input[28],                                  
           'working_status':     						                                input[29],                                              
           'days_since_last_inbound_lead_date':     				                input[30],                                  
           'days_since_last_congress_lead_date':     				                input[31],                                  
           'has_been_recommended':     						                          input[32],                                    
           'postal_code':     			 				                                input[33],                                        
           'count_clients_with_same_zipcode':     				                  input[34],                                
           'is_city_with_other_clients':     					                      input[35],                                   
           'count_clients_with_same_zipcode_and_spe':     			            input[36],                                
           'count_clients_with_same_specialty':     				                input[37],                                    
           'is_in_dense_area_for_this_cluster':     				                input[38],                                    
           'number_of_prospects_in_account':     				                    input[39],                                
           'number_of_clients_in_account':     			 		                    input[40],                               
         }                             

      ),
      contentType: false,
      cache: false,
      processData: false,
      success: function (response) {
        $('#response-container').show();
        console.log(response);
        var predict = JSON.parse(response)
        $("#response").text(predict.probability[0]);
      },
      error: function (xhr, resp, text) {
        let response = JSON.parse(xhr.responseText);
        console.log(xhr, resp, text);
      }
    });
  });
  

});
