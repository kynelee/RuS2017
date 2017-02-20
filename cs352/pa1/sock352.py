# this pseudo-code shows an example strategy for implementing
# the CS 352 socket library 

import binascii
import socket as syssock
import struct
import sys
import time
import random

# this init function is global clto the class and
# defines the UDP ports all messages are sent
# and received from.
def init(UDPportTx,UDPportRx):   # initialize your UDP socket here
    # create a UDP/datagram socket 
    global rx_port
    global rx_socket
    global tx_port

    rx_port = int(UDPportRx)
    rx_socket = syssock.socket(syssock.AF_INET, syssock.SOCK_DGRAM)
    rx_socket.bind(('', rx_port))
    tx_port = int(UDPportTx)
    
class socket:
    
    def __init__(self):
        self.seq_num = -1 # sequence number for packet which has been received last
        self.ack_num = -1
        self.connected = False # if handshake completed, set to true
        self.current_buffer = None
        self.client_closed = False
        self.packet_format = struct.Struct('!BBHQQL')


    def bind(self,address):
      # null function for part 1 
        return 

    def connect(self,address):  # fill in your code here
        global tx_port

        init_sequence_no = random.randint(1,100) 
        init_packet = self.packet_format.pack(1, 1, 24, init_sequence_no, 0, 0)
        
        while True:
          try:
            rx_socket.settimeout(.2)
            rx_socket.sendto(init_packet, ('localhost', tx_port))

            print("Sending init packet with sequence number + " + str(init_sequence_no))
            print(tx_port)

            ack = rx_socket.recv(4096) # Received ack from server
            ack = self.packet_format.unpack(ack)

            print("Received ack packet " + str(ack))

            client_ack = self.packet_format.pack(1, 0, 24, ack[4], int(ack[3]) + 1, 0) # Send final client ack
            rx_socket.sendto(client_ack, ('localhost', tx_port))

            print("Sending final client ack " + str((1, 0, 24, ack[4], int(ack[3]) + 1, 0)))

            self.seq_num = ack[4]
            self.ack_num = ack[3]
            rx_socket.settimeout(None)

            print("connected")
            break

          except syssock.timeout:
            continue

    
    def listen(self,backlog):
        return

    def accept(self):
        self.server = True
        while True:
          self.__sock352_get_packet()
          if self.connected == True:
            return (self, tx_port)
          
  
    def close(self):   # fill in your code here
        # send a FIN packet (flags with FIN bit set)
        # remove the connection from the list of connections
        
        while True:
          try:
            rx_socket.settimeout(.2)
            final_packet = self.packet_format.pack(1, 2, 0, 0, 0, 0)
            rx_socket.sendto(final_packet, ('localhost', tx_port))

            final_ack = rx_socket.recv(1096)
            final_ack = self.packet_format.unpack(final_ack)
            
            
            if final_ack[1] == 2:
              print("Terminating connection")
              rx_socket.close()
              return

          except syssock.timeout:
            continue



    def send(self, buffer):
        bytessent = 0     # fill in your code here
        payload_len = len(buffer) 

        packet = self.packet_format.pack(1, 0, 24, self.seq_num, self.ack_num, payload_len)
        packet = packet + buffer
        

        while True:
          try:
            print("sending packet number " + str(self.seq_num))
            rx_socket.sendto(packet, ('localhost', tx_port))
            rx_socket.settimeout(.2)

            ack = rx_socket.recv(1096)
            ack = self.packet_format.unpack(ack)
            if ack[4] == self.seq_num:
              self.seq_num = self.seq_num + 1
              self.ack_num = ack[3] + 1
              rx_socket.settimeout(None)
              break

          except syssock.timeout:
            continue


        return payload_len


    def recv(self,bytes_to_receive):
        if self.current_buffer: # Already data, just need to return in 
          if bytes_to_receive > len(self.current_buffer):
            data = self.current_buffer
            self.current_buffer = None
          else:           
            data = self.current_buffer[0:bytes_to_receive]
            self.current_buffer = self.current_buffer[bytes_to_receive + 1:]
            
          return data 

        else:
          self.__sock352_get_packet() # Get another packet
          
          data = self.current_buffer[0:bytes_to_receive]
          self.current_buffer = self.current_buffer[bytes_to_receive:len(self.current_buffer)]

          return data
          # now our buffer should be filled, so call again
          #return self.recv(bytes_to_receive)
         

       
    def  __sock352_get_packet(self):
        global tx_port
      
        packet = rx_socket.recv(64000)
        header = self.packet_format.unpack(packet[:24])
        data = packet[24: len(packet)]

        payload_len = len(data) 
        

        flags = header[1]

        if flags == 1: # initial handshake
          # send back ack
          print("Received connection initiation " + str(packet))

          server_seq_num = random.randint(1,100)
          ack = self.packet_format.pack(1, 0, 24, server_seq_num, header[3] + 1, 0)

          while True:
            try:
              rx_socket.settimeout(.2)
              rx_socket.sendto(ack, ('localhost', tx_port))

              print("sending server ack" + str(self.packet_format.unpack(ack)))

              client_ack = rx_socket.recv(4096)
               
              client_ack = self.packet_format.unpack(client_ack)

              print("received client ack" + str(client_ack))

              self.connected = True
              self.seq_num = int(client_ack[3])
              self.ack_num = int(client_ack[4])


              print("connected and waiting for " + str(self.seq_num))
              rx_socket.settimeout(None)
              return

            except syssock.timeout:
              continue

          print "connection established"

        if flags == 2: # closing connection
          while True:
            try:
              rx_socket.settimeout(.2)
              self.client_closed = True
              final_ack = self.packet_format.pack(1, 2, 0, 0, 0, 0)
              rx_socket.sendto(final_ack, ('localhost', tx_port))
              break

            except syssock.timeout:
              continue

          


        else:
          seq_num = header[3]
          ack_num = header[4]
          payload = data 
          if (seq_num != self.seq_num) or payload_len != header[5]: # if ack is dropped or packet is malformed
            reset_packet = self.packer_format.pack(1, 8, 24, ack_num, self.seq_num, 0)
            rx_port.sendto(reset_packet, ('localhost', tx_port))


            # send reset (RST) packet with sequence nubmer


          else:
            print("sending ack for packet " + str(self.seq_num))
            self.current_buffer = data
            self.seq_num = seq_num + 1
            # send ack with ack = seq_num + 1, seq_num = ack + 1
            ack = self.packet_format.pack(1, 0, 24, ack_num + 1, seq_num, 0)
            rx_socket.sendto(ack ,('localhost', tx_port))