import binascii
import socket as syssock
import struct
import sys
import random

# these functions are global to the class and
# define the UDP ports all messages are sent
# and received from

def init(UDPportTx,UDPportRx):   # initialize your UDP socket here  
    global tx
    global rx
    tx = int(UDPportTx)
    rx = int(UDPportRx)
    
class socket:
    
    def __init__(self):  # fill in your code here 
        tx_sock= syssock.socket(syssock.AF_INET, syssock.SOCK_STREAM)
        rx_sock = syssock.socket(syssock.AF_INET, syssock.SOCK_STREAM)
        self.tx = tx_sock
        self.rx = rx_sock
        
    def bind(self,address):
        print(tx, rx)
        self.tx.bind(('localhost', tx))
        self.rx.bind(('localhost', rx))

    def connect(self,address):  # fill in your code here 
        init_struct = struct.Struct("!BBHQQL") # version, flags, header_len, sqeuence_no, ack_no, payload_len
        init_packet= init_struct.pack(1, 1, 24, random.randint(1, 100), 0, 0)
        
        self.tx.connect(('localhost', tx))
        self.rx.connect(('localhost', rx))

        self.tx.sendall(init_packet)
        print(" Client sending init_packet = " + str(init_packet))

        while True:
          amount_received = 0
          amount_expected = len(init_packet)
          response_packet = []

          while(amount_received < amount_expected):
            chunk = self.rx.recv(8)
            response_packet.append(chunk)
            amount_received += len(chunk)
            response = "".join(response_packet)
            
        response = init_struct.unpack(response) 
        flag = response[1]

        print("Client received response packet" + str(response))

        if flag != 5: # SYN and ACK are set
          print("Existing Connection")

    
    def listen(self,backlog):
        print("Listenting!!!!")
        self.rx.listen(backlog)
        self.tx.listen(backlog)

    def accept(self):
        while True:
          clientsocket, address = self.rx.accept()
          struct_format= struct.Struct('!BBHQQL') # version, flags, header_len, sqeuence_no, ack_no, payload_len
          client_packet= struct_format.pack(1, 1, 24, random.randint(1,100), 0, 0)

          amount_received = 0
          amount_expected = len(client_packet)
          init_packet = []
          
          print("got here")

          while(amount_received < amount_expected):
            chunk = self.rx.recv(8)
            init_packet.append(chunk)
            amount_received += len(chunk)

          response = "".join(init_packet)
          response = struct_format.unpack(response)

          print("Server received init packet " + str(response))

          response[1] = 5
          response[3] = random.randInt(1, 100)
          response[4] = response[3] + 1

          print("Server sending response packet " + str(response))
          self.tx.sendall(struct_format.pack(response))
          return(clientsocket, address)


# TODO implement check for whether connection is open
          return (clientsocket,address)
    
    def close(self):   # fill in your code here 
        return 

    def send(self,buffer):
        bytessent = 0     # fill in your code here 
        return bytesent 

    def recv(self,nbytes):
        bytesreceived = 0     # fill in your code here
        return bytesreceived 


    


