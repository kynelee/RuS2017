
import binascii
import socket as syssock
import struct
import sys
import time
import random

# encryption libraries 
import nacl.utils
import nacl.secret
import nacl.utils
from nacl.public import PrivateKey, Box

# if you want to debug and print the current stack frame 
from inspect import currentframe, getframeinfo

# these are globals to the sock352 class and
# define the UDP ports all messages are sent
# and received from

# the ports to use for the sock352 messages 
global sock352portTx
global sock352portRx
# the public and private keychains in hex format 
global publicKeysHex
global privateKeysHex

# the public and private keychains in binary format 
global publicKeys
global privateKeys

# the encryption flag 
global ENCRYPT

publicKeysHex = {} 
privateKeysHex = {} 
publicKeys = {} 
privateKeys = {}

# this is 0xEC 
ENCRYPT = 236 

# this is the structure of the sock352 packet 
sock352HdrStructStr = '!BBBBHHLLQQLL'


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

def readKeyChain(filename):
    global publicKeysHex
    global privateKeysHex 
    global publicKeys
    global privateKeys 
    
    if (filename):
        try:
            keyfile_fd = open(filename,"r")
            for line in keyfile_fd:
                words = line.split()
                # check if a comment
                # more than 2 words, and the first word does not have a
                # hash, we may have a valid host/key pair in the keychain
                if ( (len(words) >= 4) and (words[0].find("#") == -1)):
                    host = words[1]
                    port = words[2]
                    keyInHex = words[3]
                    if (words[0] == "private"):
                        privateKeysHex[(host,port)] = keyInHex
                        privateKeys[(host,port)] = nacl.public.PrivateKey(keyInHex, nacl.encoding.HexEncoder)
                    elif (words[0] == "public"):
                        publicKeysHex[(host,port)] = keyInHex
                        publicKeys[(host,port)] = nacl.public.PublicKey(keyInHex, nacl.encoding.HexEncoder)
        except Exception,e:
            print ( "error: opening keychain file: %s %s" % (filename,repr(e)))
    else:
            print ("error: No filename presented")             

    return (publicKeys,privateKeys)

    
class socket:
    
    def __init__(self):
        self.seq_num = -1 # sequence number for packet which has been received last
        self.ack_num = -1
        self.connected = False # if handshake completed, set to true
        self.current_buffer = None
        self.client_closed = False
        self.packet_format = struct.Struct('!BBHQQL')
        self.address = None 
        self.client = False
        self.server = False
        self.encrypt = False
        self.encrypt_key = None
        self.decrypt_key = None 

    def bind(self,address):
      # null function for part 1 
        return 

    def connect(self,*args):  # fill in your code here
        # Check publicKeys and privateKeys to check for matching host and
        # port
        # Create nonce
        # Find Keys
        # Create Box Object:

        global ENCRYPT
        global rx_port
        global tx_port
        
        address = args[0]
        self.address = address[0] 
        self.client = True

        if (len(args) >=1): # No Encrpytion
          init_sequence_no = random.randint(1,100) 
          init_packet = self.packet_format.pack(1, 1, 24, init_sequence_no, 0, 0)
          
          while True:
            try:
              rx_socket.settimeout(.2)
              rx_socket.sendto(init_packet, (self.address, tx_port))

              print("Sending init packet with sequence number + " + str(init_sequence_no))
              print(tx_port)

              ack = rx_socket.recv(4096) # Received ack from server
              ack = self.packet_format.unpack(ack)

              print("Received ack packet " + str(ack))

              client_ack = self.packet_format.pack(1, 0, 24, ack[4], int(ack[3]) + 1, 0) # Send final client ack
              rx_socket.sendto(client_ack, (self.address, tx_port))

              print("Sending final client ack " + str((1, 0, 24, ack[4], int(ack[3]) + 1, 0)))

              self.seq_num = ack[4]
              self.ack_num = ack[3]
              rx_socket.settimeout(None)

              print("connected")
              break

            except syssock.timeout:
              continue


        elif (len(args) >=2):  
          if (args[0] == ENCRYPT):
            self.encrypt = True


          # rx port and address represents client, address represents server
        
          encrypt_key = privateKeys[('localhost', rx_port)] # retrieve clients private key used to encrypt messages 
          decrypt_key = publicKeys[address]


      










          






      

    
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


        if self.client_closed == False and self.server == True:
          while self.client_closed == False:
            self.__sock352_get_packet()
          self.close()


        if self.client:
          while True:
            try:
              rx_socket.settimeout(.2)
              final_packet = self.packet_format.pack(1, 2, 0, 0, 0, 0)
              rx_socket.sendto(final_packet, (self.address, tx_port))

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
            rx_socket.sendto(packet, (self.address, tx_port))
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
            self.current_buffer = self.current_buffer[bytes_to_receive:]
            
          return data 

        else:
          
          while not self.current_buffer and self.client_closed == False:
            try:
              print("executing")
              rx_socket.settimeout(.2)
              self.__sock352_get_packet() # Get another packet
            except syssock.timeout:
              continue

            rx_socket.settimeout(None)
          


          data = self.current_buffer[0:bytes_to_receive]
          self.current_buffer = self.current_buffer[bytes_to_receive:len(self.current_buffer)]

          return data
         

       
    def  __sock352_get_packet(self):
        global tx_port
      
        packet, address = rx_socket.recvfrom(64000)
        header = self.packet_format.unpack(packet[:24])
        data = packet[24: len(packet)]
        self.address = address[0]
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
              rx_socket.sendto(ack, address)

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
              final_ack = self.packet_format.pack(1, 2, 0, 0, 0, 0)
              rx_socket.sendto(final_ack, address)
              self.client_closed = True
              break

            except syssock.error:
              self.client_closed = True

            except syssock.timeout:
              continue


        else:
          seq_num = header[3]
          ack_num = header[4]
          payload = data 
          if (seq_num != self.seq_num) or payload_len != header[5]: # if ack is dropped or packet is malformed
            print("Seq num = " + str(seq_num) )
            print("self.seq_num = " + str(self.seq_num))

            self.seq_num = seq_num + 1
            reset_packet = self.packet_format.pack(1, 8, 24, ack_num, seq_num, 0)
            rx_socket.sendto(reset_packet, address)
            # send reset (RST) packet with sequence nubmer

          else:
            print("sending ack for packet " + str(self.seq_num))
            self.current_buffer = data
            self.seq_num = seq_num + 1
            # send ack with ack = seq_num + 1, seq_num = ack + 1
            ack = self.packet_format.pack(1, 0, 24, ack_num + 1, seq_num, 0)
            if (random.randint(1, 10) > 8):
              print("dropping ack")
              
            else:
              rx_socket.sendto(ack , address)
                
